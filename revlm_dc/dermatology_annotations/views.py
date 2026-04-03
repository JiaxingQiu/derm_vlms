import json
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render

from .models import Annotation, Dermatologist


def load_valid_users():
    json_path = Path(settings.BASE_DIR) / "data" / "users.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)["users"]


def load_annotations_data():
    json_path = Path(settings.BASE_DIR) / "data" / "annotations_data.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_case_model_data(case_data):
    return {
        key: value
        for key, value in case_data.items()
        if key != "image_path" and isinstance(value, dict)
    }


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

    if "difficulty" not in payload and request.POST.get("difficulty"):
        payload["difficulty"] = request.POST.get("difficulty")

    if "other_feedback" not in payload and request.POST.get("textual_feedback"):
        payload["other_feedback"] = request.POST.get("textual_feedback", "").strip()

    if "visual_feedback" not in payload and request.POST.get("visual_feedback"):
        payload["visual_feedback"] = request.POST.get("visual_feedback", "").strip()

    if "model_response_correct" not in payload and "model_response_correct" in request.POST:
        payload["model_response_correct"] = request.POST.get("model_response_correct") == "on"

    return payload


def normalize_item_feedback(items):
    normalized = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        label = item.get("label") or item.get("status") or ""
        label = {
            "yes": "correct",
            "no": "incorrect",
            "correct": "correct",
            "incorrect": "incorrect",
            "maybe": "maybe",
        }.get(label, "")
        normalized.append(
            {
                "text": item.get("text", ""),
                "label": label,
                "feedback": item.get("feedback", ""),
                "crops": item.get("crops", []),
            }
        )
    return normalized


def normalize_model_review(model_review, existing_model_review=None):
    existing_model_review = existing_model_review or {}

    difficulty = model_review.get("difficulty", existing_model_review.get("difficulty"))
    if difficulty in ("", None):
        difficulty = None
    elif isinstance(difficulty, str) and difficulty.isdigit():
        difficulty = int(difficulty)

    raw_state = model_review.get("raw_state", existing_model_review.get("raw_state", {}))

    normalized = {
        "diagnosis_feedback": normalize_item_feedback(
            model_review.get("diagnosis_feedback", existing_model_review.get("diagnosis_feedback", []))
        ),
        "description_feedback": normalize_item_feedback(
            model_review.get("description_feedback", existing_model_review.get("description_feedback", []))
        ),
        "other_feedback": model_review.get(
            "other_feedback",
            existing_model_review.get("other_feedback", ""),
        ),
        "other_feedback_crops": model_review.get(
            "other_feedback_crops",
            existing_model_review.get("other_feedback_crops", []),
        ),
        "difficulty": difficulty,
        "raw_state": raw_state if isinstance(raw_state, dict) else {},
    }

    if "visual_feedback" in model_review:
        normalized["visual_feedback"] = model_review.get("visual_feedback", "")
    elif existing_model_review.get("visual_feedback"):
        normalized["visual_feedback"] = existing_model_review["visual_feedback"]

    return normalized


def summarize_text_feedback(review):
    parts = []

    for item in review.get("diagnosis_feedback", []):
        if item.get("label") == "incorrect" and item.get("feedback"):
            parts.append(f"Diagnosis correction: {item['feedback']}")

    for item in review.get("description_feedback", []):
        if item.get("label") == "incorrect" and item.get("feedback"):
            parts.append(f"Description correction: {item['feedback']}")

    other_feedback = review.get("other_feedback", "").strip()
    if other_feedback:
        parts.append(other_feedback)

    return "\n".join(parts) if parts else ""


def summarize_visual_feedback(review):
    visual_payload = {}

    for key in ("other_feedback_crops", "visual_feedback"):
        value = review.get(key)
        if value:
            visual_payload[key] = value

    diag_crops = [
        item["crops"]
        for item in review.get("diagnosis_feedback", [])
        if item.get("crops")
    ]
    desc_crops = [
        item["crops"]
        for item in review.get("description_feedback", [])
        if item.get("crops")
    ]

    if diag_crops:
        visual_payload["diagnosis_crops"] = diag_crops
    if desc_crops:
        visual_payload["description_crops"] = desc_crops

    return json.dumps(visual_payload) if visual_payload else ""


def review_is_fully_correct(review):
    labels = [
        item.get("label")
        for group in ("diagnosis_feedback", "description_feedback")
        for item in review.get(group, [])
        if item.get("label")
    ]
    return bool(labels) and all(label == "correct" for label in labels)


def build_frontend_review(payload, existing_review=None):
    existing_review = existing_review or {}
    existing_models_review = dict(existing_review.get("models", {}))

    if isinstance(payload.get("models"), dict):
        models_review = {}
        for model_name, model_review in payload["models"].items():
            if not isinstance(model_review, dict):
                continue
            models_review[model_name] = normalize_model_review(
                model_review,
                existing_models_review.get(model_name, {}),
            )

        return {
            "active_model": payload.get("active_model") or existing_review.get("active_model"),
            "models": models_review,
        }

    model_name = payload.get("model_name") or payload.get("vlm") or payload.get("model") or "default"
    models_review = dict(existing_models_review)
    models_review[model_name] = normalize_model_review(
        payload,
        existing_models_review.get(model_name, {}),
    )
    return {"active_model": model_name, "models": models_review}


def update_annotation_from_frontend(annotation, payload):
    review_data = build_frontend_review(payload, annotation.review_data)
    active_model = review_data["active_model"]
    active_review = review_data["models"][active_model]

    annotation.review_data = review_data
    annotation.difficulty = active_review.get("difficulty")
    annotation.model_response_correct = review_is_fully_correct(active_review)
    annotation.textual_feedback = summarize_text_feedback(active_review)
    annotation.visual_feedback = summarize_visual_feedback(active_review)


def apply_legacy_form_payload(annotation, request):
    annotation.model_response_correct = request.POST.get("model_response_correct") == "on"
    annotation.textual_feedback = request.POST.get("textual_feedback", "").strip()
    annotation.visual_feedback = request.POST.get("visual_feedback", "").strip()


def login_view(request):
    error_message = None

    if request.method == "POST":
        login_id = request.POST.get("login_id", "").strip()
        valid_users = load_valid_users()

        if login_id not in valid_users:
            error_message = "invalid login id"
        else:
            Dermatologist.objects.get_or_create(login_id=login_id)
            request.session["login_id"] = login_id
            return redirect("annotations")

    return render(request, "login.html", {"error_message": error_message})


def get_model_keys(case_data):
    return [k for k in case_data if k != "image_path" and isinstance(case_data[k], dict)]


def build_page_sequence(case_ids, annotations_data):
    pages = []
    for case_id in case_ids:
        case_data = annotations_data.get(case_id, {})
        model_keys = get_model_keys(case_data)
        for model_key in model_keys:
            pages.append((case_id, model_key))
    return pages


def is_page_complete(review_data, model_key, case_data):
    model_review = (review_data or {}).get("models", {}).get(model_key, {})
    source = case_data.get(model_key, {})
    total = len(source.get("diagnoses", [])) + len(source.get("descriptions", []))
    if total == 0:
        return True
    reviewed = 0
    for item in model_review.get("diagnosis_feedback", []):
        if item.get("label"):
            reviewed += 1
    for item in model_review.get("description_feedback", []):
        if item.get("label"):
            reviewed += 1
    return reviewed >= total


def find_first_incomplete_page(pages, annotations_data, dermatologist):
    annotations_cache = {}
    for pi, (case_id, model_key) in enumerate(pages):
        if case_id not in annotations_cache:
            try:
                ann = Annotation.objects.get(dermatologist=dermatologist, case_id=case_id)
                annotations_cache[case_id] = ann.review_data
            except Annotation.DoesNotExist:
                annotations_cache[case_id] = {}
        case_data = annotations_data.get(case_id, {})
        if not is_page_complete(annotations_cache[case_id], model_key, case_data):
            return pi
    return len(pages)


def annotations_view(request):
    login_id = request.session.get("login_id")
    if not login_id:
        return redirect("login")

    annotations_data = load_annotations_data()
    dermatologist = get_object_or_404(Dermatologist, login_id=login_id)
    case_ids = sorted(annotations_data.keys())

    if not case_ids:
        return render(
            request,
            "annotations.html",
            {
                "login_id": login_id,
                "no_cases": True,
            },
        )

    pages = build_page_sequence(case_ids, annotations_data)
    total_pages = len(pages)

    if request.method == "GET":
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

    if flat_index >= total_pages:
        if request.method == "POST":
            action = request.POST.get("action")

            if action == "done_yes":
                dermatologist.is_done = True
                dermatologist.save()
                return redirect("thank_you")

            if action == "done_no":
                flat_index = total_pages - 1
                case_id, model_key = pages[flat_index]
                ci = case_ids.index(case_id)
                case_data = annotations_data[case_id]
                mi = get_model_keys(case_data).index(model_key)
                dermatologist.current_case_index = ci
                dermatologist.current_model_index = mi
                dermatologist.save()
                return redirect("annotations")

        return render(
            request,
            "annotations.html",
            {
                "login_id": login_id,
                "show_done_confirmation": True,
            },
        )

    if dermatologist.is_done:
        return redirect("thank_you")

    flat_index = min(flat_index, total_pages - 1)
    current_case_id, current_model_key = pages[flat_index]
    current_case_data = annotations_data[current_case_id]
    current_model_data = current_case_data[current_model_key]

    annotation, _ = Annotation.objects.get_or_create(
        dermatologist=dermatologist,
        case_id=current_case_id,
    )

    if request.method == "POST":
        payload = parse_request_payload(request)
        action = payload.get("action", "save")

        payload["model_name"] = current_model_key
        if (
            is_json_request(request)
            or isinstance(payload.get("models"), dict)
            or payload.get("active_model")
            or payload.get("model_name")
            or payload.get("diagnosis_feedback")
            or payload.get("description_feedback")
            or payload.get("other_feedback") is not None
        ):
            update_annotation_from_frontend(annotation, payload)
        else:
            apply_legacy_form_payload(annotation, request)

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
            nav_mi = get_model_keys(annotations_data[nav_case_id]).index(nav_model_key)
            dermatologist.current_case_index = nav_ci
            dermatologist.current_model_index = nav_mi

        dermatologist.save()

        if is_json_request(request):
            return JsonResponse(
                {
                    "ok": True,
                    "action": action,
                    "case_id": current_case_id,
                    "model_key": current_model_key,
                    "current_page": new_flat,
                    "annotation_id": annotation.id,
                    "saved_review_data": annotation.review_data,
                }
            )

        return redirect("annotations")

    saved_model_review = (annotation.review_data or {}).get("models", {}).get(current_model_key, {})

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
    }

    return render(request, "annotations.html", context)


def thank_you_view(request):
    login_id = request.session.get("login_id")
    if not login_id:
        return redirect("login")

    return render(request, "thank_you.html", {"login_id": login_id})


def logout_view(request):
    request.session.flush()
    return redirect("login")
