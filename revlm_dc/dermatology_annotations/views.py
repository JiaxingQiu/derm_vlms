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

from .models import Annotation, Dermatologist, TabAuthSession


_EMPTY_TC = {"text": "", "crops": []}
AUTH_TOKEN_PARAM = "auth"
AUTH_TOKEN_TTL = timedelta(hours=12)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_users_config():
    json_path = Path(settings.BASE_DIR) / "data" / "users.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_valid_users():
    return load_users_config()["users"]


def load_annotations_data():
    json_path = Path(settings.BASE_DIR) / "data" / "annotations_data.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# User / assignment helpers
# ---------------------------------------------------------------------------

def get_user_case_ids(users_config, login_id):
    """Return the ordered list of case_ids assigned to this user."""
    user_data = users_config["users"].get(login_id, {})
    assignments = user_data.get("assignments", [])
    return [a["case_id"] for a in assignments]


def get_case_interface_map(users_config, login_id):
    """Return {case_id: interface_type} for this user's assignments."""
    user_data = users_config["users"].get(login_id, {})
    assignments = user_data.get("assignments", [])
    return {
        a["case_id"]: a.get("conditions", {}).get("interface_type", "conditional")
        for a in assignments
    }


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
    "maybe": "maybe",
}


def normalize_item_feedback(items):
    """Normalise a list of per-item feedback dicts.

    Each item keeps {text, label, feedback, crops} with crops stored as
    a per-item array of {x, y, w, h} rectangles so the UI can re-render
    evidence thumbnails on reload.
    """
    normalised = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        label = _LABEL_MAP.get(item.get("label") or item.get("status") or "", "")
        normalised.append({
            "text": item.get("text", ""),
            "label": label,
            "feedback": item.get("feedback", ""),
            "crops": item.get("crops", []),
        })
    return normalised


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

    annotation.model = model_key
    annotation.interface_type = "conditional"
    annotation.raw_response = model_info.get("raw_response", "")

    annotation.diagnosis_feedback = normalize_item_feedback(
        payload.get("diagnosis_feedback", annotation.diagnosis_feedback or []),
    )
    annotation.description_feedback = normalize_item_feedback(
        payload.get("description_feedback", annotation.description_feedback or []),
    )

    other_text = payload.get("other_feedback", "")
    other_crops = payload.get("other_feedback_crops", [])
    if other_text or other_crops:
        annotation.other_feedback = _tc(other_text, other_crops)
    elif not (annotation.other_feedback or {}).get("text"):
        annotation.other_feedback = dict(_EMPTY_TC)


def update_annotation_unconditional(annotation, payload):
    """Populate *unconditional* (human-only) fields on an annotation."""
    annotation.interface_type = "unconditional"
    annotation.model = ""

    existing = {
        "user_diagnoses": [
            (annotation.user_diagnosis_1 or _EMPTY_TC).get("text", ""),
            (annotation.user_diagnosis_2 or _EMPTY_TC).get("text", ""),
            (annotation.user_diagnosis_3 or _EMPTY_TC).get("text", ""),
        ],
        "user_diagnoses_crops": [
            (annotation.user_diagnosis_1 or _EMPTY_TC).get("crops", []),
            (annotation.user_diagnosis_2 or _EMPTY_TC).get("crops", []),
            (annotation.user_diagnosis_3 or _EMPTY_TC).get("crops", []),
        ],
        "user_reasons": (annotation.user_reasons or _EMPTY_TC).get("text", ""),
        "user_reasons_crops": (annotation.user_reasons or _EMPTY_TC).get("crops", []),
    }

    diags = payload.get("user_diagnoses", existing["user_diagnoses"])
    diag_crops = payload.get("user_diagnoses_crops", existing["user_diagnoses_crops"])

    annotation.user_diagnosis_1 = _tc(
        diags[0] if len(diags) > 0 else "",
        diag_crops[0] if len(diag_crops) > 0 else [],
    )
    annotation.user_diagnosis_2 = _tc(
        diags[1] if len(diags) > 1 else "",
        diag_crops[1] if len(diag_crops) > 1 else [],
    )
    annotation.user_diagnosis_3 = _tc(
        diags[2] if len(diags) > 2 else "",
        diag_crops[2] if len(diag_crops) > 2 else [],
    )
    annotation.user_reasons = _tc(
        payload.get("user_reasons", existing["user_reasons"]),
        payload.get("user_reasons_crops", existing["user_reasons_crops"]),
    )


# ---------------------------------------------------------------------------
# Page sequence / completion
# ---------------------------------------------------------------------------

def build_page_sequence(case_ids, annotations_data, interface_map=None):
    pages = []
    for case_id in case_ids:
        iface = (interface_map or {}).get(case_id, "conditional")
        if iface == "unconditional":
            pages.append((case_id, None))
        else:
            case_data = annotations_data.get(case_id, {})
            model_keys = get_model_keys(case_data)
            for model_key in model_keys:
                pages.append((case_id, model_key))
    return pages


def is_page_complete(annotation, model_key, case_data):
    """Check whether an annotation page has all required fields filled."""
    if annotation is None:
        return False

    if model_key is None:
        diags = [
            (annotation.user_diagnosis_1 or _EMPTY_TC).get("text", ""),
            (annotation.user_diagnosis_2 or _EMPTY_TC).get("text", ""),
            (annotation.user_diagnosis_3 or _EMPTY_TC).get("text", ""),
        ]
        all_diags = all(d.strip() for d in diags)
        has_reasons = bool(
            (annotation.user_reasons or _EMPTY_TC).get("text", "").strip()
        )
        return all_diags and has_reasons

    source = case_data.get(model_key, {})
    total = len(source.get("diagnoses", [])) + len(source.get("descriptions", []))
    if total == 0:
        return True

    all_items = list(annotation.diagnosis_feedback or []) + list(annotation.description_feedback or [])
    reviewed = sum(1 for item in all_items if item.get("label"))
    for item in all_items:
        if item.get("label") == "incorrect" and not (item.get("feedback") or "").strip():
            return False
    return reviewed >= total


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

    if request.method == "POST":
        login_id = request.POST.get("login_id", "").strip()
        valid_users = load_valid_users()

        if login_id not in valid_users:
            error_message = "invalid login id"
        else:
            raw_token, _ = create_tab_auth_session(login_id)
            return redirect(auth_url("annotations", raw_token))

    return render(request, "login.html", {"error_message": error_message})


TEMPLATE_MAP = {
    "conditional": "annotations_conditional.html",
    "unconditional": "annotations_unconditional.html",
}


@never_cache
@csrf_exempt
def annotations_view(request):
    raw_token, tab_session = get_tab_auth_session(request)
    if tab_session is None:
        return redirect("login")

    login_id = tab_session.dermatologist.login_id

    annotations_data = load_annotations_data()
    users_config = load_users_config()
    dermatologist = tab_session.dermatologist

    user_case_ids = get_user_case_ids(users_config, login_id)

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

    interface_map = get_case_interface_map(users_config, login_id)
    pages = build_page_sequence(case_ids, annotations_data, interface_map)
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
                return redirect(auth_url("thank_you", raw_token))

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
        return redirect(auth_url("thank_you", raw_token))

    # ---- Current page data ----
    flat_index = min(flat_index, total_pages - 1)
    current_case_id, current_model_key = pages[flat_index]
    current_case_data = annotations_data[current_case_id]
    current_interface = interface_map.get(current_case_id, "conditional")

    current_model_data = (
        current_case_data.get(current_model_key, {})
        if current_model_key is not None
        else {}
    )

    annotation, _ = Annotation.objects.get_or_create(
        dermatologist=dermatologist,
        case_id=current_case_id,
        model=current_model_key or "",
        defaults={"interface_type": current_interface},
    )

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

        if current_interface == "unconditional":
            update_annotation_unconditional(annotation, payload)
        else:
            update_annotation_conditional(
                annotation, payload, current_model_key, current_case_data,
            )

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
            if nav_model_key is None:
                nav_mi = 0
            else:
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
    if current_interface == "unconditional":
        d1 = annotation.user_diagnosis_1 or _EMPTY_TC
        d2 = annotation.user_diagnosis_2 or _EMPTY_TC
        d3 = annotation.user_diagnosis_3 or _EMPTY_TC
        ur = annotation.user_reasons or _EMPTY_TC
        saved_unconditional = {
            "user_diagnoses": [d1.get("text", ""), d2.get("text", ""), d3.get("text", "")],
            "user_diagnoses_crops": [d1.get("crops", []), d2.get("crops", []), d3.get("crops", [])],
            "user_reasons": ur.get("text", ""),
            "user_reasons_crops": ur.get("crops", []),
        }
        saved_model_review = {}
    else:
        of = annotation.other_feedback or _EMPTY_TC
        saved_unconditional = {}
        saved_model_review = {
            "diagnosis_feedback": annotation.diagnosis_feedback or [],
            "description_feedback": annotation.description_feedback or [],
            "other_feedback": of.get("text", ""),
            "other_feedback_crops": of.get("crops", []),
        }

    template = TEMPLATE_MAP.get(current_interface, "annotations_conditional.html")

    context = {
        "login_id": login_id,
        "case_id": current_case_id,
        "case_data": current_case_data,
        "model_key": current_model_key,
        "model_data": current_model_data,
        "interface_type": current_interface,
        "annotation": annotation,
        "saved_model_review": saved_model_review,
        "saved_unconditional": saved_unconditional,
        "current_page": flat_index,
        "total_pages": total_pages,
        "has_previous": flat_index > 0,
        "has_next": flat_index < total_pages - 1,
        "auth_token": raw_token,
    }

    return render(request, template, context)


@never_cache
@csrf_exempt
def thank_you_view(request):
    raw_token, tab_session = get_tab_auth_session(request)
    if tab_session is None:
        return redirect("login")
    return render(
        request,
        "thank_you.html",
        {"login_id": tab_session.dermatologist.login_id, "auth_token": raw_token},
    )


@never_cache
@csrf_exempt
def logout_view(request):
    revoke_tab_auth_session(extract_auth_token(request))
    return redirect("login")
