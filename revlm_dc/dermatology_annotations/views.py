import json
from pathlib import Path
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from .models import Dermatologist, Annotation

### helpers
def load_users_data():
    json_path = Path(settings.BASE_DIR) / "data" / "users.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)["users"]

def load_annotations_data():
    json_path = Path(settings.BASE_DIR) / "data" / "annotations_data.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

### views

# def dermatology_annotations(request):
#     dermatologists = Dermatologist.objects.all()
#     return render(request, "annotations.html", {"dermatologists": dermatologists})

def login_view(request):
    error_message = None

    if request.method == "POST":
        login_id = request.POST.get("login_id", "").strip()
        data = load_users_data() # NOTE: if too slow, use separate json

        if login_id not in data:
            error_message = "invalid login id"
        else:
            dermatologist, created = Dermatologist.objects.get_or_create(
                login_id=login_id
            )
            request.session["login_id"] = login_id
            return redirect("annotations")

    return render(request, "login.html", {"error_message": error_message})


def annotations_view(request):
    # validate login
    login_id = request.session.get("login_id")
    if not login_id:
        return redirect("login")

    # load cases data
    data = load_annotations_data()
    dermatologist = get_object_or_404(Dermatologist, login_id=login_id)

    # get current case
    user_cases_dict = data[login_id]
    case_ids = list(user_cases_dict.keys())

    # TODO: send to thank you page if all cases annotated
    if dermatologist.current_case_index >= len(case_ids):
        if request.method == "POST":
            action = request.POST.get("action")

            if action == "done_yes":
                dermatologist.is_done = True
                dermatologist.save()
                return redirect("thank_you")

            if action == "done_no":
                dermatologist.current_case_index = len(case_ids) - 1
                dermatologist.save()
                return redirect("annotations")

        return render(request, "annotations.html", {
            "login_id": login_id,
            "show_done_confirmation": True,
        })

    if dermatologist.is_done:
        return redirect("thank_you")

    current_index = dermatologist.current_case_index
    current_case_id = case_ids[current_index]
    current_case_data = user_cases_dict[current_case_id]

    annotation, created = Annotation.objects.get_or_create(
        dermatologist=dermatologist,
        case_id=current_case_id
    )

    # if request.method == "POST":
    #     action = request.POST.get("action")

    #     annotation.model_response_correct = request.POST.get("model_response_correct") == "on"
    #     annotation.textual_feedback = request.POST.get("textual_feedback", "").strip()
    #     annotation.visual_feedback = request.POST.get("visual_feedback", "").strip()
    #     annotation.save()

    #     if action == "next" and current_index < len(case_ids) - 1:
    #         dermatologist.current_case_index = current_index + 1
    #         dermatologist.save()
    #         return redirect("annotations")

    #     if action == "previous" and current_index > 0:
    #         dermatologist.current_case_index = current_index - 1
    #         dermatologist.save()
    #         return redirect("annotations")

    #     return redirect("annotations")

    if request.method == "POST":
        action = request.POST.get("action")

        annotation.model_response_correct = request.POST.get("model_response_correct") == "on"
        annotation.textual_feedback = request.POST.get("textual_feedback", "").strip()
        annotation.visual_feedback = request.POST.get("visual_feedback", "").strip()
        annotation.save()

        if action == "previous" and current_index > 0:
            dermatologist.current_case_index = current_index - 1
            dermatologist.save()
            return redirect("annotations")

        if action == "next" and current_index < len(case_ids) - 1:
            dermatologist.current_case_index = current_index + 1
            dermatologist.save()
            return redirect("annotations")

        if action == "finish":
            dermatologist.current_case_index = len(case_ids)
            dermatologist.save()
            return redirect("annotations")

        return redirect("annotations")

    context = {
        "login_id": login_id,
        "case_id": current_case_id,
        "case_data": current_case_data,
        "annotation": annotation,
        "current_index": current_index,
        "total_cases": len(case_ids),
        "has_previous": current_index > 0,
        "has_next": current_index < len(case_ids) - 1,
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

