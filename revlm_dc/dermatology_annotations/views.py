import json
import hashlib
import re
import secrets
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlencode

from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.db import transaction
from django.http import JsonResponse
from django.urls import reverse
from django.shortcuts import redirect, render
from django.utils import timezone

from .models import (
    Annotation, Assignment, Dermatologist,
    PCPAnnotation, PCPAssignment, PCPUser,
    TabAuthSession,
)


_EMPTY_TC = {"text": "", "crops": []}
AUTH_TOKEN_PARAM = "auth"
AUTH_TOKEN_TTL = timedelta(hours=12)
AUTH_IDLE_TIMEOUT = timedelta(hours=2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_annotations_data():
    json_path = Path(settings.BASE_DIR) / "data" / "annotations_data.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# User / assignment helpers (DB-backed)
# ---------------------------------------------------------------------------

def get_user_case_ids(evaluator, role="Dermatologist"):
    """Return the ordered list of case_ids assigned to this evaluator."""
    Model = PCPAssignment if role == "PCP" else Assignment
    return list(
        Model.objects
        .filter(evaluator=evaluator)
        .order_by("order")
        .values_list("case_id", flat=True)
    )


def get_model_keys(case_data, case_id=None):
    """Return model keys for a case, deterministically shuffled per lesion."""
    keys = [k for k in case_data if k != "image_path" and isinstance(case_data[k], dict)]
    if case_id is not None:
        from .assignments import DEFAULT_SEED, _stable_hash
        lesion_id = case_id.rsplit("_", 1)[0]
        keys.sort(key=lambda k: _stable_hash(DEFAULT_SEED, "model_order", lesion_id, k))
    return keys


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------

def is_json_request(request):
    return "application/json" in (request.content_type or "")


def parse_request_payload(request):
    if is_json_request(request):
        try:
            return json.loads(request.body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return {}

    review_data = request.POST.get("review_data")
    if review_data:
        try:
            payload = json.loads(review_data)
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}

    if request.POST.get("action"):
        payload["action"] = request.POST.get("action")

    if request.POST.get("target_page") is not None:
        try:
            payload["target_page"] = int(request.POST.get("target_page"))
        except (ValueError, TypeError):
            pass

    return payload


def hash_auth_token(raw_token):
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def create_tab_auth_session(login_id, role="Dermatologist"):
    raw_token = secrets.token_urlsafe(32)
    now = timezone.now()
    if role == "PCP":
        pcp_user, _ = PCPUser.objects.get_or_create(login_id=login_id)
        TabAuthSession.objects.filter(
            pcp_user=pcp_user, revoked_at__isnull=True,
        ).update(revoked_at=now)
        TabAuthSession.objects.create(
            pcp_user=pcp_user,
            token_hash=hash_auth_token(raw_token),
            expires_at=now + AUTH_TOKEN_TTL,
        )
        return raw_token, pcp_user
    else:
        dermatologist, _ = Dermatologist.objects.get_or_create(login_id=login_id)
        TabAuthSession.objects.filter(
            dermatologist=dermatologist, revoked_at__isnull=True,
        ).update(revoked_at=now)
        TabAuthSession.objects.create(
            dermatologist=dermatologist,
            token_hash=hash_auth_token(raw_token),
            expires_at=now + AUTH_TOKEN_TTL,
        )
        return raw_token, dermatologist


def extract_auth_token(request):
    auth_header = request.headers.get("Authorization", "").strip()
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    return (
        request.POST.get(AUTH_TOKEN_PARAM, "").strip()
        or request.GET.get(AUTH_TOKEN_PARAM, "").strip()
        or request.headers.get("X-Annotation-Auth", "").strip()
    )


def get_tab_auth_session(request):
    raw_token = extract_auth_token(request)
    if not raw_token:
        return "", None

    now = timezone.now()
    tab_session = (
        TabAuthSession.objects.select_related("dermatologist", "pcp_user")
        .filter(
            token_hash=hash_auth_token(raw_token),
            revoked_at__isnull=True,
            expires_at__gt=now,
            last_used_at__gt=now - AUTH_IDLE_TIMEOUT,
        )
        .first()
    )
    if tab_session is None:
        return raw_token, None

    TabAuthSession.objects.filter(pk=tab_session.pk).update(last_used_at=timezone.now())
    return raw_token, tab_session


def revoke_tab_auth_session(raw_token):
    if not raw_token:
        return

    TabAuthSession.objects.filter(
        token_hash=hash_auth_token(raw_token),
        revoked_at__isnull=True,
    ).update(revoked_at=timezone.now())


def auth_url(view_name, raw_token, **query_params):
    params = {AUTH_TOKEN_PARAM: raw_token}
    params.update(query_params)
    return f"{reverse(view_name)}?{urlencode(params)}"


# ---------------------------------------------------------------------------
# Feedback normalisation
# ---------------------------------------------------------------------------

_LABEL_MAP = {
    "yes": "correct",
    "no": "incorrect",
    "correct": "correct",
    "incorrect": "incorrect",
}


def normalize_reasoning_edits(edits, ai_sentences):
    """Align user edits with the AI's reasoning sentences.

    Produces one entry per AI sentence: {original, edited, crops}. ``edited``
    equals ``original`` when the user did not touch the sentence. ``crops``
    is a list of {x, y, w, h} rectangles anchored to the case image.

    User-added entries (original == "added") beyond the AI sentence count
    are preserved at the end of the list.
    """
    out = []
    edits = edits or []
    for idx, original in enumerate(ai_sentences or []):
        entry = edits[idx] if idx < len(edits) and isinstance(edits[idx], dict) else {}
        edited = entry.get("edited")
        if edited is None:
            edited = entry.get("text", original)
        result = {
            "original": str(original),
            "edited": str(edited if edited is not None else original),
            "crops": entry.get("crops") if isinstance(entry.get("crops"), list) else [],
        }
        for key in ("ev0_clinical", "ev0_dscope",
                     "ev0_clinical_original", "ev0_dscope_original"):
            val = entry.get(key)
            if val is not None and isinstance(val, dict):
                result[key] = val
            elif key in entry:
                result[key] = None
        if entry.get("_ev0_moved"):
            result["_ev0_moved"] = True
        out.append(result)

    # Preserve user-added reasoning entries beyond AI sentences
    for idx in range(len(ai_sentences or []), len(edits)):
        entry = edits[idx]
        if isinstance(entry, dict) and entry.get("original") == "added":
            out.append({
                "original": "added",
                "edited": str(entry.get("edited", "") or ""),
                "crops": entry.get("crops") if isinstance(entry.get("crops"), list) else [],
            })
    return out


def _split_review_to_fields(items):
    """Translate the canonical list-of-3 review shape (the wire format
    used by the template / autosave POSTs) into the per-field storage
    shape used on ``Annotation`` (``diagnosis_N`` + ``reasoning_N``).

    Missing slots are materialised as empty dict / empty list so the six
    fields always have a uniform shape.
    """
    items = items or []
    out = {}
    for k in range(3):
        item = items[k] if k < len(items) and isinstance(items[k], dict) else {}
        out[f"diagnosis_{k + 1}"] = {
            "name": item.get("name", "") or "",
            "label": item.get("label", "") or "",
            "correct_differential": item.get("correct_differential", "") or "",
        }
        out[f"reasoning_{k + 1}"] = list(item.get("reasoning_edits", []) or [])
    return out


def _merge_review_from_fields(annotation):
    """Inverse of :func:`_split_review_to_fields`: rebuild the list-of-3
    wire shape from the six per-diagnosis fields on an Annotation.

    Empty trailing slots (no name and no reasoning) are dropped so the
    template renders only the diagnoses the AI actually produced.
    """
    merged = []
    for k in range(1, 4):
        d = getattr(annotation, f"diagnosis_{k}", None) or {}
        r = getattr(annotation, f"reasoning_{k}", None) or []
        if not d and not r:
            continue
        merged.append({
            "name": d.get("name", ""),
            "label": d.get("label", ""),
            "reasoning_edits": list(r),
            "correct_differential": d.get("correct_differential", ""),
        })
    return merged


def normalize_diagnosis_feedback(items, ai_diagnoses):
    """Normalise the per-diagnosis review payload.

    Each returned entry:
        {
            "name": str,                       # AI diagnosis name (preserved)
            "label": "" | "correct" | "incorrect",
            "reasoning_edits": [{"original": str, "edited": str, "crops": [...]}, ...],
            "correct_differential": str,       # human replacement diagnosis name
        }

    When a user deletes a diagnosis and provides a replacement:
    - label is set to "incorrect"
    - correct_differential holds the human's replacement diagnosis name
    - reasoning_edits[0] has original="deleted" with edited=human reasoning

    ``ai_diagnoses`` is the list coming from the annotations_data.json
    (``[{name, reasoning_sentences}]``) so we can always lock the schema
    to what the AI produced and avoid drift between UI edits and source.
    """
    items = items or []
    by_idx = {idx: item for idx, item in enumerate(items) if isinstance(item, dict)}

    out = []
    for idx, ai_d in enumerate(ai_diagnoses or []):
        raw = by_idx.get(idx, {})
        raw_label = raw.get("label") or ""
        label = _LABEL_MAP.get(raw_label, "")
        raw_edits = raw.get("reasoning_edits") or []

        # Detect prepended "deleted" sentinel and pass it through as-is
        has_replacement = (
            len(raw_edits) > 0
            and isinstance(raw_edits[0], dict)
            and raw_edits[0].get("original") == "deleted"
        )
        if has_replacement:
            raw_rep_crops = raw_edits[0].get("crops", [])
            replacement_entry = {
                "original": "deleted",
                "edited": str(raw_edits[0].get("edited", "") or ""),
                "crops": raw_rep_crops if isinstance(raw_rep_crops, list) else [],
            }
            ai_edits = raw_edits[1:]
        else:
            replacement_entry = None
            ai_edits = raw_edits

        reasoning_edits = normalize_reasoning_edits(
            ai_edits,
            ai_d.get("reasoning_sentences", []),
        )

        if replacement_entry:
            reasoning_edits = [replacement_entry] + reasoning_edits

        out.append({
            "name": ai_d.get("name", raw.get("name", "")),
            "label": label,
            "reasoning_edits": reasoning_edits,
            "correct_differential": str(raw.get("correct_differential", "") or "").strip(),
        })
    return out


def _tc(text="", crops=None):
    """Build a {text, crops} dict."""
    return {"text": text, "crops": crops if crops is not None else []}


def _get_tc(source, key, default=None):
    """Read a {text, crops} value from a dict or return existing default."""
    val = source.get(key)
    if isinstance(val, dict) and "text" in val:
        return {"text": val.get("text", ""), "crops": val.get("crops", [])}
    if default is not None:
        return default
    return dict(_EMPTY_TC)


# ---------------------------------------------------------------------------
# Annotation update helpers
# ---------------------------------------------------------------------------

def update_annotation_conditional(annotation, payload, model_key, case_data):
    """Populate *conditional* (AI evaluation) fields on an annotation."""
    model_info = case_data.get(model_key, {})
    ai_diagnoses = model_info.get("diagnoses", [])

    annotation.model = model_key
    annotation.raw_response = model_info.get("raw_response", "")

    incoming = payload.get(
        "diagnosis_feedback",
        _merge_review_from_fields(annotation),
    )
    normalized = normalize_diagnosis_feedback(incoming, ai_diagnoses)
    for field, value in _split_review_to_fields(normalized).items():
        setattr(annotation, field, value)

    other_text = payload.get("other_feedback", "")
    other_crops = payload.get("other_feedback_crops", [])
    if "other_feedback" in payload or "other_feedback_crops" in payload:
        annotation.other_feedback = _tc(other_text, other_crops)
    elif not (annotation.other_feedback or {}).get("text"):
        annotation.other_feedback = dict(_EMPTY_TC)

    raw_order = payload.get("diagnosis_order")
    if isinstance(raw_order, list) and all(isinstance(x, int) for x in raw_order):
        annotation.diagnosis_order = raw_order


# ---------------------------------------------------------------------------
# Page sequence / completion
# ---------------------------------------------------------------------------

def build_page_sequence(case_ids, annotations_data):
    """Expand case_ids into (case_id, model_key) page tuples."""
    pages = []
    for case_id in case_ids:
        case_data = annotations_data.get(case_id, {})
        model_keys = get_model_keys(case_data, case_id)
        for model_key in model_keys:
            pages.append((case_id, model_key))
    return pages


def build_page_list(pages, annotations_data, completion_map):
    """Build a JSON-serialisable list for the sidebar navigation.

    ``completion_map`` maps ``(case_id, model_key_or_empty)`` → bool.
    """
    out = []
    for i, (case_id, model_key) in enumerate(pages):
        lesion_id = case_id.rsplit("_", 1)[0] if "_" in case_id else case_id
        case_data = annotations_data.get(case_id, {})
        model_keys = get_model_keys(case_data, case_id)
        model_num = model_keys.index(model_key) + 1 if model_key in model_keys else 1
        out.append({
            "idx": i,
            "lesion": lesion_id,
            "model": model_num,
            "done": completion_map.get((case_id, model_key or ""), False),
        })
    return out


def is_page_complete(annotation, model_key, case_data):
    """Check whether an annotation page has all required fields filled."""
    if annotation is None:
        return False

    # Unconditional (human-only) pages are retired and persist no data.
    # If one is ever rendered (the template + routing are still present),
    # it has nothing to gate on, so treat it as auto-complete.
    if model_key is None:
        return True

    source = case_data.get(model_key, {})
    ai_diagnoses = source.get("diagnoses", [])
    if not ai_diagnoses:
        return True

    feedback = _merge_review_from_fields(annotation)
    if len(feedback) < len(ai_diagnoses):
        return False

    return True


def find_first_incomplete_page(pages, annotations_data, evaluator,
                               annotation_model=Annotation,
                               evaluator_field="dermatologist"):
    cache = {}
    for pi, (case_id, model_key) in enumerate(pages):
        cache_key = (case_id, model_key or "")
        if cache_key not in cache:
            try:
                cache[cache_key] = annotation_model.objects.get(
                    **{evaluator_field: evaluator},
                    case_id=case_id,
                    model=model_key or "",
                )
            except annotation_model.DoesNotExist:
                cache[cache_key] = None
        case_data = annotations_data.get(case_id, {})
        if not is_page_complete(cache[cache_key], model_key, case_data):
            return pi
    return len(pages)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def _login_id_exists(login_id):
    """Check if a login_id is taken in either user table."""
    return (Dermatologist.objects.filter(login_id=login_id).exists()
            or PCPUser.objects.filter(login_id=login_id).exists())


def _find_existing_user(login_id):
    """Return (user_obj, role_str) or (None, None)."""
    try:
        return Dermatologist.objects.get(login_id=login_id), "Dermatologist"
    except Dermatologist.DoesNotExist:
        pass
    try:
        return PCPUser.objects.get(login_id=login_id), "PCP"
    except PCPUser.DoesNotExist:
        pass
    return None, None


@never_cache
@csrf_exempt
def login_view(request):
    raw_token, tab_session = get_tab_auth_session(request)
    if tab_session is not None:
        return redirect(auth_url("annotations", raw_token))

    error_message = None
    success_message = None
    show_register = False
    form_data = {}

    if request.method == "POST":
        action = request.POST.get("action", "login")
        login_id = request.POST.get("login_id", "").strip().lower()
        form_data["login_id"] = login_id

        if action == "register":
            show_register = True
            full_name = request.POST.get("full_name", "").strip()
            occupation = request.POST.get("occupation", "").strip()
            years_experience = request.POST.get("years_experience", "").strip()
            institution = request.POST.get("institution", "").strip()
            dermoscopy_experience = request.POST.get("dermoscopy_experience", "").strip()
            zip_code = request.POST.get("zip_code", "").strip()
            form_data.update({
                "full_name": full_name,
                "occupation": occupation,
                "years_experience": years_experience,
                "institution": institution,
                "dermoscopy_experience": dermoscopy_experience,
                "zip_code": zip_code,
            })

            if not login_id or not full_name or not occupation or not years_experience or not institution or not dermoscopy_experience or not zip_code:
                error_message = "All fields are required."
            elif not years_experience.isdigit() or not (0 <= int(years_experience) <= 100):
                error_message = "Years at this speciality must be a whole number between 0 and 100."
            elif _login_id_exists(login_id):
                error_message = "Username already taken."
            else:
                from .assignments import next_available_slot, assign_from_slot
                role = "PCP" if occupation == "PCP" else "Dermatologist"
                UserModel = PCPUser if role == "PCP" else Dermatologist
                LockModel = UserModel

                with transaction.atomic():
                    _lock_qs = (LockModel.objects
                                .select_for_update()
                                .order_by("pk")[:1])
                    list(_lock_qs)

                    slot = next_available_slot(role=role)
                    evaluator = UserModel.objects.create(
                        login_id=login_id,
                        full_name=full_name,
                        occupation=occupation,
                        years_experience=int(years_experience),
                        institution=institution,
                        dermoscopy_experience=dermoscopy_experience,
                        zip_code=zip_code,
                        assignment_slot=slot,
                    )
                    assign_from_slot(evaluator, slot, role=role)
                raw_token, _ = create_tab_auth_session(login_id, role=role)
                return redirect(auth_url("annotations", raw_token))

        else:
            if not login_id:
                error_message = "Please enter your username."
            elif login_id == "test":
                from .assignments import assign_from_slot
                evaluator, created = Dermatologist.objects.get_or_create(
                    login_id="test",
                    defaults={"full_name": "Test User", "occupation": "Tester", "institution": "Demo"},
                )
                assign_from_slot(evaluator, slot_number=0, role="Dermatologist")
                Annotation.objects.filter(dermatologist=evaluator).delete()
                evaluator.current_case_index = 0
                evaluator.current_model_index = 0
                evaluator.is_done = False
                evaluator.save()
                raw_token, _ = create_tab_auth_session(login_id, role="Dermatologist")
                return redirect(auth_url("annotations", raw_token))
            elif login_id == "test_pcp":
                from .assignments import assign_from_slot
                evaluator, created = PCPUser.objects.get_or_create(
                    login_id="test_pcp",
                    defaults={"full_name": "Test PCP User", "occupation": "PCP", "institution": "Demo"},
                )
                assign_from_slot(evaluator, slot_number=0, role="PCP")
                PCPAnnotation.objects.filter(pcp_user=evaluator).delete()
                evaluator.current_case_index = 0
                evaluator.current_model_index = 0
                evaluator.is_done = False
                evaluator.save()
                raw_token, _ = create_tab_auth_session(login_id, role="PCP")
                return redirect(auth_url("annotations", raw_token))
            else:
                user_obj, role = _find_existing_user(login_id)
                if user_obj is None:
                    error_message = "Username not found. Please register first."
                else:
                    raw_token, _ = create_tab_auth_session(login_id, role=role)
                    return redirect(auth_url("annotations", raw_token))

    return render(request, "login.html", {
        "error_message": error_message,
        "success_message": success_message,
        "show_register": show_register,
        "form_data": form_data,
    })


@never_cache
def demo_login_view(request):
    """Static shareable link that lands straight on the annotations page as "test".

    Mints a fresh token per visit, so the /demo/ URL itself never goes stale
    the way a copied ``?auth=`` link would.
    """
    from .assignments import assign_from_slot

    evaluator, _ = Dermatologist.objects.get_or_create(
        login_id="test",
        defaults={"full_name": "Test User", "occupation": "Tester", "institution": "Demo"},
    )
    assign_from_slot(evaluator, slot_number=0, role="Dermatologist")
    Annotation.objects.filter(dermatologist=evaluator).delete()
    evaluator.current_case_index = 0
    evaluator.current_model_index = 0
    evaluator.is_done = False
    evaluator.save()

    raw_token, _ = create_tab_auth_session("test", role="Dermatologist")
    return redirect(auth_url("annotations", raw_token))


@never_cache
@csrf_exempt
def session_check_view(request):
    """Lightweight endpoint for the client to verify its token is still valid."""
    _, tab_session = get_tab_auth_session(request)
    if tab_session is None:
        return JsonResponse({"ok": False, "reason": "session_expired"}, status=401)
    return JsonResponse({"ok": True})


@never_cache
@csrf_exempt
def annotations_view(request):
    raw_token, tab_session = get_tab_auth_session(request)
    if tab_session is None:
        if is_json_request(request):
            return JsonResponse(
                {"ok": False, "reason": "session_expired"}, status=401,
            )
        return redirect("login")

    if tab_session.pcp_user_id:
        return _pcp_annotations_view(request, raw_token, tab_session)

    return _derm_annotations_view(request, raw_token, tab_session)


def _derm_annotations_view(request, raw_token, tab_session):
    """Annotation view for Dermatologist users — all pages are conditional."""
    login_id = tab_session.dermatologist.login_id
    dermatologist = tab_session.dermatologist

    annotations_data = load_annotations_data()
    user_case_ids = get_user_case_ids(dermatologist, role="Dermatologist")

    if user_case_ids:
        case_ids = [cid for cid in user_case_ids if cid in annotations_data]
    else:
        def natural_key(s):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]
        case_ids = sorted(annotations_data.keys(), key=natural_key)

    if not case_ids:
        return render(
            request,
            "annotations_conditional.html",
            {"login_id": login_id, "no_cases": True, "auth_token": raw_token},
        )

    pages = build_page_sequence(case_ids, annotations_data)
    total_pages = len(pages)

    # ---- Determine current flat page index ----
    current_case_idx = dermatologist.current_case_index
    current_model_idx = dermatologist.current_model_index
    if current_case_idx >= len(case_ids):
        flat_index = total_pages
    else:
        target_case_id = case_ids[current_case_idx]
        flat_index = 0
        for pi, (pid, _) in enumerate(pages):
            if pid == target_case_id:
                flat_index = pi + current_model_idx
                break

    # ---- Done-confirmation screen ----
    if flat_index >= total_pages:
        if request.method == "POST":
            action = request.POST.get("action")

            if action == "done_yes":
                dermatologist.is_done = True
                dermatologist.save()
                revoke_tab_auth_session(raw_token)
                return redirect("login")

            if action == "done_no":
                flat_index = total_pages - 1
                case_id, model_key = pages[flat_index]
                ci = case_ids.index(case_id)
                if model_key is None:
                    mi = 0
                else:
                    case_data = annotations_data[case_id]
                    mi = get_model_keys(case_data, case_id).index(model_key)
                dermatologist.current_case_index = ci
                dermatologist.current_model_index = mi
                dermatologist.save()
                return redirect(auth_url("annotations", raw_token, nav=1))

        return render(
            request,
            "annotations_conditional.html",
            {
                "login_id": login_id,
                "show_done_confirmation": True,
                "auth_token": raw_token,
            },
        )

    if dermatologist.is_done:
        revoke_tab_auth_session(raw_token)
        return redirect("login")

    # ---- Current page data ----
    flat_index = min(flat_index, total_pages - 1)
    current_case_id, current_model_key = pages[flat_index]
    current_case_data = annotations_data[current_case_id]

    current_model_data = (
        current_case_data.get(current_model_key, {})
        if current_model_key is not None
        else {}
    )

    annotation, _ = Annotation.objects.get_or_create(
        dermatologist=dermatologist,
        case_id=current_case_id,
        model=current_model_key or "",
    )

    now = timezone.now()
    visits = annotation.page_visits or []
    if request.method == "GET":
        if not visits or visits[-1]["completed_at"] is not None:
            visits.append({"entered_at": now.isoformat(), "completed_at": None})
            annotation.page_visits = visits
            annotation.save(update_fields=["page_visits"])

    # ---- POST: save annotation ----
    if request.method == "POST":
        payload = parse_request_payload(request)
        action = payload.get("action", "save")

        if action == "reset_all":
            Annotation.objects.filter(dermatologist=dermatologist).delete()
            dermatologist.current_case_index = 0
            dermatologist.current_model_index = 0
            dermatologist.is_done = False
            dermatologist.save()
            return redirect(auth_url("annotations", raw_token))

        page_cid = payload.pop("_page_case_id", None)
        page_mk = payload.pop("_page_model_key", None)
        if page_cid is not None and (
            page_cid != current_case_id or page_mk != current_model_key
        ):
            if is_json_request(request):
                return JsonResponse({"ok": False, "reason": "stale_page"})
            return redirect(auth_url("annotations", raw_token, nav=1))

        if action == "mark_complete":
            annotation.marked_complete = payload.get("mark_complete_value") == "true"
            annotation.save(update_fields=["marked_complete", "updated_at"])
            if is_json_request(request):
                return JsonResponse({"ok": True, "action": "mark_complete"})
            return redirect(auth_url("annotations", raw_token, nav=1))

        if action == "goto":
            target = payload.get("target_page")
            if isinstance(target, int) and 0 <= target < total_pages:
                nav_case_id, nav_model_key = pages[target]
                nav_ci = case_ids.index(nav_case_id)
                nav_mi = get_model_keys(annotations_data[nav_case_id], nav_case_id).index(nav_model_key)
                dermatologist.current_case_index = nav_ci
                dermatologist.current_model_index = nav_mi
                dermatologist.save()
            return redirect(auth_url("annotations", raw_token, nav=1))

        update_annotation_conditional(
            annotation, payload, current_model_key, current_case_data,
        )

        now_ts = timezone.now().isoformat()
        visits = annotation.page_visits or []
        if visits:
            visits[-1]["completed_at"] = now_ts
            annotation.page_visits = visits

        annotation.save()

        new_flat = flat_index
        if action == "previous" and flat_index > 0:
            new_flat = flat_index - 1
        elif action == "next" and flat_index < total_pages - 1:
            new_flat = flat_index + 1
        elif action == "finish":
            new_flat = total_pages

        if new_flat >= total_pages:
            dermatologist.current_case_index = len(case_ids)
            dermatologist.current_model_index = 0
        else:
            nav_case_id, nav_model_key = pages[new_flat]
            nav_ci = case_ids.index(nav_case_id)
            nav_mi = get_model_keys(annotations_data[nav_case_id], nav_case_id).index(nav_model_key)
            dermatologist.current_case_index = nav_ci
            dermatologist.current_model_index = nav_mi

        dermatologist.save()

        if is_json_request(request):
            return JsonResponse({
                "ok": True,
                "action": action,
                "case_id": current_case_id,
                "model_key": current_model_key,
                "current_page": new_flat,
                "annotation_id": annotation.id,
                "redirect_url": auth_url("annotations", raw_token, nav=1),
            })

        return redirect(auth_url("annotations", raw_token, nav=1))

    # ---- GET: build template context ----
    of = annotation.other_feedback or _EMPTY_TC
    saved_model_review = {
        "diagnosis_feedback": _merge_review_from_fields(annotation),
        "diagnosis_order": annotation.diagnosis_order or [],
        "other_feedback": of.get("text", ""),
        "other_feedback_crops": of.get("crops", []),
    }

    all_model_keys = get_model_keys(current_case_data, current_case_id)
    model_num = all_model_keys.index(current_model_key) + 1 if current_model_key in all_model_keys else 1
    model_display_name = "AI Model " + str(model_num)

    lesion_id = current_case_id.rsplit("_", 1)[0] if "_" in current_case_id else current_case_id
    lesion_display_name = f"Lesion {lesion_id}"

    completion_map = {
        (a.case_id, a.model): a.marked_complete
        for a in Annotation.objects.filter(dermatologist=dermatologist)
                                   .only("case_id", "model", "marked_complete")
    }
    page_list = build_page_list(pages, annotations_data, completion_map)

    context = {
        "login_id": login_id,
        "case_id": current_case_id,
        "lesion_display_name": lesion_display_name,
        "case_data": current_case_data,
        "model_key": current_model_key,
        "model_display_name": model_display_name,
        "model_data": current_model_data,
        "annotation": annotation,
        "saved_model_review": saved_model_review,
        "current_page": flat_index,
        "total_pages": total_pages,
        "has_previous": flat_index > 0,
        "has_next": flat_index < total_pages - 1,
        "marked_complete": annotation.marked_complete,
        "page_list": page_list,
        "auth_token": raw_token,
    }

    return render(request, "annotations_conditional.html", context)


# ---------------------------------------------------------------------------
# PCP annotation view — first half unconditional, second half conditional
# ---------------------------------------------------------------------------

def _build_pcp_page_sequence(case_ids, annotations_data):
    """Build PCP page sequence: first half unconditional, second half conditional.

    Returns list of (case_id, model_key_or_None, interface_type) tuples.
    model_key is None for unconditional pages.
    """
    half = len(case_ids) // 2
    uncond_ids = case_ids[:half]
    cond_ids = case_ids[half:]

    pages = []
    for case_id in uncond_ids:
        pages.append((case_id, None, "unconditional"))

    for case_id in cond_ids:
        case_data = annotations_data.get(case_id, {})
        model_keys = get_model_keys(case_data, case_id)
        for model_key in model_keys:
            pages.append((case_id, model_key, "conditional"))

    return pages


def _pcp_annotations_view(request, raw_token, tab_session):
    """Annotation view for PCP users."""
    pcp_user = tab_session.pcp_user
    login_id = pcp_user.login_id

    annotations_data = load_annotations_data()
    user_case_ids = get_user_case_ids(pcp_user, role="PCP")

    if user_case_ids:
        case_ids = [cid for cid in user_case_ids if cid in annotations_data]
    else:
        def natural_key(s):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]
        case_ids = sorted(annotations_data.keys(), key=natural_key)

    if not case_ids:
        return render(
            request,
            "annotations_conditional.html",
            {"login_id": login_id, "no_cases": True, "auth_token": raw_token},
        )

    pages = _build_pcp_page_sequence(case_ids, annotations_data)
    total_pages = len(pages)

    # ---- Determine current flat page index ----
    current_case_idx = pcp_user.current_case_index
    current_model_idx = pcp_user.current_model_index
    if current_case_idx >= len(case_ids):
        flat_index = total_pages
    else:
        target_case_id = case_ids[current_case_idx]
        flat_index = 0
        for pi, (pid, _, _) in enumerate(pages):
            if pid == target_case_id:
                flat_index = pi + current_model_idx
                break

    # ---- Done-confirmation screen ----
    if flat_index >= total_pages:
        if request.method == "POST":
            action = request.POST.get("action")

            if action == "done_yes":
                pcp_user.is_done = True
                pcp_user.save()
                revoke_tab_auth_session(raw_token)
                return redirect("login")

            if action == "done_no":
                flat_index = total_pages - 1
                case_id, model_key, itype = pages[flat_index]
                ci = case_ids.index(case_id)
                if model_key is None:
                    mi = 0
                else:
                    case_data = annotations_data[case_id]
                    mi = get_model_keys(case_data, case_id).index(model_key)
                pcp_user.current_case_index = ci
                pcp_user.current_model_index = mi
                pcp_user.save()
                return redirect(auth_url("annotations", raw_token, nav=1))

        return render(
            request,
            "annotations_conditional.html",
            {
                "login_id": login_id,
                "show_done_confirmation": True,
                "auth_token": raw_token,
            },
        )

    if pcp_user.is_done:
        revoke_tab_auth_session(raw_token)
        return redirect("login")

    # ---- Current page data ----
    flat_index = min(flat_index, total_pages - 1)
    current_case_id, current_model_key, current_itype = pages[flat_index]
    current_case_data = annotations_data[current_case_id]

    annotation, _ = PCPAnnotation.objects.get_or_create(
        pcp_user=pcp_user,
        case_id=current_case_id,
        model=current_model_key or "",
        defaults={"interface_type": current_itype},
    )

    now = timezone.now()
    visits = annotation.page_visits or []
    if request.method == "GET":
        if not visits or visits[-1]["completed_at"] is not None:
            visits.append({"entered_at": now.isoformat(), "completed_at": None})
            annotation.page_visits = visits
            annotation.save(update_fields=["page_visits"])

    # ---- POST: save annotation ----
    if request.method == "POST":
        payload = parse_request_payload(request)
        action = payload.get("action", "save")

        if action == "reset_all":
            PCPAnnotation.objects.filter(pcp_user=pcp_user).delete()
            pcp_user.current_case_index = 0
            pcp_user.current_model_index = 0
            pcp_user.is_done = False
            pcp_user.save()
            return redirect(auth_url("annotations", raw_token))

        page_cid = payload.pop("_page_case_id", None)
        page_mk = payload.pop("_page_model_key", None)
        if page_cid is not None and (
            page_cid != current_case_id or (page_mk or "") != (current_model_key or "")
        ):
            if is_json_request(request):
                return JsonResponse({"ok": False, "reason": "stale_page"})
            return redirect(auth_url("annotations", raw_token, nav=1))

        if action == "mark_complete":
            annotation.marked_complete = payload.get("mark_complete_value") == "true"
            annotation.save(update_fields=["marked_complete", "updated_at"])
            if is_json_request(request):
                return JsonResponse({"ok": True, "action": "mark_complete"})
            return redirect(auth_url("annotations", raw_token, nav=1))

        if action == "goto":
            target = payload.get("target_page")
            if isinstance(target, int) and 0 <= target < total_pages:
                nav_case_id, nav_model_key, _ = pages[target]
                nav_ci = case_ids.index(nav_case_id)
                if nav_model_key is None:
                    nav_mi = 0
                else:
                    nav_mi = get_model_keys(
                        annotations_data[nav_case_id], nav_case_id,
                    ).index(nav_model_key)
                pcp_user.current_case_index = nav_ci
                pcp_user.current_model_index = nav_mi
                pcp_user.save()
            return redirect(auth_url("annotations", raw_token, nav=1))

        if current_itype == "unconditional":
            annotation.interface_type = "unconditional"
            user_diagnoses = payload.get("user_diagnoses", [])
            user_reasons = payload.get("user_reasons", [])
            user_reasons_crops = payload.get("user_reasons_crops", [])
            for i in range(3):
                dx = user_diagnoses[i] if i < len(user_diagnoses) else ""
                sentences = user_reasons[i] if i < len(user_reasons) and isinstance(user_reasons[i], list) else []
                sent_crops = user_reasons_crops[i] if i < len(user_reasons_crops) and isinstance(user_reasons_crops[i], list) else []
                reasoning_entries = []
                for si, text in enumerate(sentences):
                    crops = sent_crops[si] if si < len(sent_crops) and isinstance(sent_crops[si], list) else []
                    reasoning_entries.append({
                        "original": text or "",
                        "edited": text or "",
                        "crops": crops,
                    })
                if not reasoning_entries:
                    reasoning_entries = [{"original": "", "edited": "", "crops": []}]
                setattr(annotation, f"diagnosis_{i + 1}", {
                    "name": dx,
                    "label": "",
                    "correct_differential": "",
                })
                setattr(annotation, f"reasoning_{i + 1}", reasoning_entries)
            other_text = payload.get("other_feedback", "")
            other_crops = payload.get("other_feedback_crops", [])
            annotation.other_feedback = _tc(other_text, other_crops)
        else:
            annotation.interface_type = "conditional"
            update_annotation_conditional(
                annotation, payload, current_model_key, current_case_data,
            )

        now_ts = timezone.now().isoformat()
        visits = annotation.page_visits or []
        if visits:
            visits[-1]["completed_at"] = now_ts
            annotation.page_visits = visits

        annotation.save()

        new_flat = flat_index
        if action == "previous" and flat_index > 0:
            new_flat = flat_index - 1
        elif action == "next" and flat_index < total_pages - 1:
            new_flat = flat_index + 1
        elif action == "finish":
            new_flat = total_pages

        if new_flat >= total_pages:
            pcp_user.current_case_index = len(case_ids)
            pcp_user.current_model_index = 0
        else:
            nav_case_id, nav_model_key, _ = pages[new_flat]
            nav_ci = case_ids.index(nav_case_id)
            if nav_model_key is None:
                nav_mi = 0
            else:
                nav_mi = get_model_keys(
                    annotations_data[nav_case_id], nav_case_id,
                ).index(nav_model_key)
            pcp_user.current_case_index = nav_ci
            pcp_user.current_model_index = nav_mi

        pcp_user.save()

        if is_json_request(request):
            return JsonResponse({
                "ok": True,
                "action": action,
                "case_id": current_case_id,
                "model_key": current_model_key or "",
                "current_page": new_flat,
                "annotation_id": annotation.id,
                "redirect_url": auth_url("annotations", raw_token, nav=1),
            })

        return redirect(auth_url("annotations", raw_token, nav=1))

    # ---- GET: build template context and choose template ----
    lesion_id = current_case_id.rsplit("_", 1)[0] if "_" in current_case_id else current_case_id
    lesion_display_name = f"Lesion {lesion_id}"

    completion_map = {
        (a.case_id, a.model): a.marked_complete
        for a in PCPAnnotation.objects.filter(pcp_user=pcp_user)
                                      .only("case_id", "model", "marked_complete")
    }
    pcp_pages_2tuple = [(cid, mk) for cid, mk, _ in pages]
    page_list = build_page_list(pcp_pages_2tuple, annotations_data, completion_map)

    if current_itype == "unconditional":
        user_diagnoses = []
        user_reasons = []
        user_reasons_crops = []
        for i in range(1, 4):
            dx = getattr(annotation, f"diagnosis_{i}", None) or {}
            user_diagnoses.append(dx.get("name", ""))
            reasoning = getattr(annotation, f"reasoning_{i}", None) or []
            sent_texts = []
            sent_crops = []
            for entry in reasoning:
                if isinstance(entry, dict):
                    sent_texts.append(entry.get("edited", ""))
                    sent_crops.append(entry.get("crops", []))
            user_reasons.append(sent_texts if sent_texts else [""])
            user_reasons_crops.append(sent_crops if sent_crops else [[]])
        of = annotation.other_feedback or _EMPTY_TC
        saved_uncond = {
            "user_diagnoses": user_diagnoses,
            "user_reasons": user_reasons,
            "user_reasons_crops": user_reasons_crops,
            "other_feedback": of.get("text", ""),
            "other_feedback_crops": of.get("crops", []),
        }
        context = {
            "login_id": login_id,
            "case_id": current_case_id,
            "lesion_display_name": lesion_display_name,
            "case_data": current_case_data,
            "saved_unconditional": saved_uncond,
            "current_page": flat_index,
            "total_pages": total_pages,
            "has_previous": flat_index > 0,
            "has_next": flat_index < total_pages - 1,
            "marked_complete": annotation.marked_complete,
            "page_list": page_list,
            "auth_token": raw_token,
        }
        return render(request, "annotations_unconditional.html", context)
    else:
        current_model_data = (
            current_case_data.get(current_model_key, {})
            if current_model_key is not None
            else {}
        )
        of = annotation.other_feedback or _EMPTY_TC
        saved_model_review = {
            "diagnosis_feedback": _merge_review_from_fields(annotation),
            "diagnosis_order": annotation.diagnosis_order or [],
            "other_feedback": of.get("text", ""),
            "other_feedback_crops": of.get("crops", []),
        }

        all_model_keys = get_model_keys(current_case_data, current_case_id)
        model_num = all_model_keys.index(current_model_key) + 1 if current_model_key in all_model_keys else 1
        model_display_name = "AI Model " + str(model_num)

        context = {
            "login_id": login_id,
            "case_id": current_case_id,
            "lesion_display_name": lesion_display_name,
            "case_data": current_case_data,
            "model_key": current_model_key,
            "model_display_name": model_display_name,
            "model_data": current_model_data,
            "annotation": annotation,
            "saved_model_review": saved_model_review,
            "current_page": flat_index,
            "total_pages": total_pages,
            "has_previous": flat_index > 0,
            "has_next": flat_index < total_pages - 1,
            "marked_complete": annotation.marked_complete,
            "page_list": page_list,
            "auth_token": raw_token,
        }
        return render(request, "annotations_conditional.html", context)


@never_cache
@csrf_exempt
def logout_view(request):
    revoke_tab_auth_session(extract_auth_token(request))
    return redirect("login")
