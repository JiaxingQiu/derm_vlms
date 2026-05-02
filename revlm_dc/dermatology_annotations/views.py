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
from django.http import JsonResponse
from django.urls import reverse
from django.shortcuts import redirect, render
from django.utils import timezone

from .models import Annotation, Assignment, Dermatologist, TabAuthSession


_EMPTY_TC = {"text": "", "crops": []}
AUTH_TOKEN_PARAM = "auth"
AUTH_TOKEN_TTL = timedelta(hours=12)


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

def get_user_case_ids(dermatologist):
    """Return the ordered list of case_ids assigned to this evaluator."""
    return list(
        Assignment.objects
        .filter(evaluator=dermatologist)
        .order_by("order")
        .values_list("case_id", flat=True)
    )


def get_model_keys(case_data):
    return [k for k in case_data if k != "image_path" and isinstance(case_data[k], dict)]


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

    return payload


def hash_auth_token(raw_token):
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def create_tab_auth_session(login_id):
    dermatologist, _ = Dermatologist.objects.get_or_create(login_id=login_id)
    raw_token = secrets.token_urlsafe(32)
    TabAuthSession.objects.create(
        dermatologist=dermatologist,
        token_hash=hash_auth_token(raw_token),
        expires_at=timezone.now() + AUTH_TOKEN_TTL,
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

    tab_session = (
        TabAuthSession.objects.select_related("dermatologist")
        .filter(
            token_hash=hash_auth_token(raw_token),
            revoked_at__isnull=True,
            expires_at__gt=timezone.now(),
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
    """
    out = []
    edits = edits or []
    for idx, original in enumerate(ai_sentences or []):
        entry = edits[idx] if idx < len(edits) and isinstance(edits[idx], dict) else {}
        edited = entry.get("edited")
        if edited is None:
            edited = entry.get("text", original)
        out.append({
            "original": str(original),
            "edited": str(edited if edited else original),
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
            "correct_differential_crops": list(
                item.get("correct_differential_crops", []) or []
            ),
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
            "correct_differential_crops": list(d.get("correct_differential_crops", []) or []),
        })
    return merged


def normalize_diagnosis_feedback(items, ai_diagnoses):
    """Normalise the per-diagnosis review payload.

    Each returned entry:
        {
            "name": str,                       # AI diagnosis name
            "label": "" | "correct" | "incorrect",
            "reasoning_edits": [{"original": str, "edited": str, "crops": [...]}, ...],
            "correct_differential": str,
            "correct_differential_crops": [{x,y,w,h}, ...],
        }
    ``ai_diagnoses`` is the list coming from the annotations_data.json
    (``[{name, reasoning_sentences}]``) so we can always lock the schema
    to what the AI produced and avoid drift between UI edits and source.
    """
    items = items or []
    by_idx = {idx: item for idx, item in enumerate(items) if isinstance(item, dict)}

    out = []
    for idx, ai_d in enumerate(ai_diagnoses or []):
        raw = by_idx.get(idx, {})
        label = _LABEL_MAP.get(raw.get("label") or "", "")
        reasoning_edits = normalize_reasoning_edits(
            raw.get("reasoning_edits"),
            ai_d.get("reasoning_sentences", []),
        )
        cd_crops = raw.get("correct_differential_crops")
        out.append({
            "name": ai_d.get("name", raw.get("name", "")),
            "label": label,
            "reasoning_edits": reasoning_edits,
            "correct_differential": str(raw.get("correct_differential", "") or "").strip(),
            "correct_differential_crops": cd_crops if isinstance(cd_crops, list) else [],
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
    if other_text or other_crops:
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
        model_keys = get_model_keys(case_data)
        for model_key in model_keys:
            pages.append((case_id, model_key))
    return pages


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

    for item in feedback[: len(ai_diagnoses)]:
        label = (item or {}).get("label") or ""
        if label not in ("correct", "incorrect"):
            return False
        if label == "incorrect" and not (item.get("correct_differential") or "").strip():
            return False
    return True


def find_first_incomplete_page(pages, annotations_data, dermatologist):
    cache = {}
    for pi, (case_id, model_key) in enumerate(pages):
        cache_key = (case_id, model_key or "")
        if cache_key not in cache:
            try:
                cache[cache_key] = Annotation.objects.get(
                    dermatologist=dermatologist,
                    case_id=case_id,
                    model=model_key or "",
                )
            except Annotation.DoesNotExist:
                cache[cache_key] = None
        case_data = annotations_data.get(case_id, {})
        if not is_page_complete(cache[cache_key], model_key, case_data):
            return pi
    return len(pages)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

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
            institution = request.POST.get("institution", "").strip()
            form_data.update({
                "full_name": full_name,
                "occupation": occupation,
                "institution": institution,
            })

            if not login_id or not full_name or not occupation or not institution:
                error_message = "All fields are required."
            elif Dermatologist.objects.filter(login_id=login_id).exists():
                error_message = "Username already taken."
            else:
                from .assignments import assign_cases_for_user
                evaluator = Dermatologist.objects.create(
                    login_id=login_id,
                    full_name=full_name,
                    occupation=occupation,
                    institution=institution,
                )
                assign_cases_for_user(evaluator)
                raw_token, _ = create_tab_auth_session(login_id)
                return redirect(auth_url("annotations", raw_token))

        else:
            if not login_id:
                error_message = "Please enter your username."
            elif login_id == "test":
                from .assignments import assign_cases_for_user
                evaluator, created = Dermatologist.objects.get_or_create(
                    login_id="test",
                    defaults={"full_name": "Test User", "occupation": "Tester", "institution": "Demo"},
                )
                if created:
                    assign_cases_for_user(evaluator)
                Annotation.objects.filter(dermatologist=evaluator).delete()
                evaluator.current_case_index = 0
                evaluator.current_model_index = 0
                evaluator.is_done = False
                evaluator.save()
                raw_token, _ = create_tab_auth_session(login_id)
                return redirect(auth_url("annotations", raw_token))
            elif not Dermatologist.objects.filter(login_id=login_id).exists():
                error_message = "Username not found. Please register first."
            else:
                raw_token, _ = create_tab_auth_session(login_id)
                return redirect(auth_url("annotations", raw_token))

    return render(request, "login.html", {
        "error_message": error_message,
        "success_message": success_message,
        "show_register": show_register,
        "form_data": form_data,
    })


@never_cache
@csrf_exempt
def annotations_view(request):
    raw_token, tab_session = get_tab_auth_session(request)
    if tab_session is None:
        return redirect("login")

    login_id = tab_session.dermatologist.login_id

    annotations_data = load_annotations_data()
    dermatologist = tab_session.dermatologist

    user_case_ids = get_user_case_ids(dermatologist)

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
    is_nav_redirect = request.method == "GET" and request.GET.get("nav") == "1"

    if request.method == "GET" and not is_nav_redirect:
        flat_index = find_first_incomplete_page(pages, annotations_data, dermatologist)
    else:
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
                    mi = get_model_keys(case_data).index(model_key)
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

    # Stamp the first time the user lands on this (case, model) page.
    # Set once; never overwritten on autosaves, navigation back-and-forth,
    # or reloads.
    if annotation.first_entered_at is None:
        annotation.first_entered_at = timezone.now()
        annotation.save(update_fields=["first_entered_at"])

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

        update_annotation_conditional(
            annotation, payload, current_model_key, current_case_data,
        )

        # Stamp the first successful Next/Finish click. Set once; never
        # overwritten if the user navigates back and re-completes.
        if action in ("next", "finish") and annotation.first_completed_at is None:
            annotation.first_completed_at = timezone.now()

        annotation.save()

        # Advance navigation
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
            nav_mi = get_model_keys(annotations_data[nav_case_id]).index(nav_model_key)
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

    context = {
        "login_id": login_id,
        "case_id": current_case_id,
        "case_data": current_case_data,
        "model_key": current_model_key,
        "model_data": current_model_data,
        "annotation": annotation,
        "saved_model_review": saved_model_review,
        "current_page": flat_index,
        "total_pages": total_pages,
        "has_previous": flat_index > 0,
        "has_next": flat_index < total_pages - 1,
        "auth_token": raw_token,
    }

    return render(request, "annotations_conditional.html", context)


@never_cache
@csrf_exempt
def logout_view(request):
    revoke_tab_auth_session(extract_auth_token(request))
    return redirect("login")
