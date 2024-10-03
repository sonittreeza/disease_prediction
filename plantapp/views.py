from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.auth.views import LoginView
from django.urls import reverse_lazy
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout
from .models import *
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from django.db.models import Count
from django.core.files.uploadedfile import SimpleUploadedFile
from .models import ImageUpload, Plant, Disease
import os
import tempfile




def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()  # This saves the new user
            messages.success(request, 'Your account has been created successfully!')
            return redirect('login')  # Redirect to login page after successful signup
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

class CustomLoginView(LoginView):
    template_name = 'login.html'  
    redirect_authenticated_user = True  
    success_url = reverse_lazy('dashboard') 


def custom_logout_view(request):
    logout(request)
    return redirect('/')



@login_required  
def dashboard_view(request):
    user = request.user  # Get the authenticated user

    # Filter data based on the authenticated user
    total_scans = ImageUpload.objects.filter(user=user).count()
    diseases_detected = Disease.objects.filter(imageupload__user=user).distinct()  # Diseases detected by the user's uploads
    recent_activity = ImageUpload.objects.filter(user=user).order_by('-uploaded_at')[:5]
    accuracy_rate = 89  # This can be calculated dynamically based on user-specific data if available

    # Disease data for the chart
    disease_labels = [disease.name for disease in diseases_detected]
    disease_detections = [ImageUpload.objects.filter(user=user, detected_disease=disease).count() for disease in diseases_detected]

    context = {
        'total_scans': total_scans,
        'diseases_detected': diseases_detected.count(),
        'accuracy_rate': accuracy_rate,
        'recent_activity': recent_activity,
        'disease_labels': disease_labels,
        'disease_detections': disease_detections,
    }
    return render(request, 'dashboard.html', context)



def home(request):
    return render(request, 'home.html')

def profile_management(request):
    return render(request, 'profile_management.html')


def report_list(request):
    user=request.user
    reports = Report.objects.filter(user=user)
    return render(request, 'detailed_disease_reports.html', {'reports': reports})

def report_detail_view(request, report_id):
    # Retrieve the report using the provided report ID
    report = get_object_or_404(Report, id=report_id)
    
    # Render the report details in the template
    return render(request, 'report_detail.html', {'report': report})

from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from .models import Report
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

def download_report(request, report_id):
    report = get_object_or_404(Report, id=report_id)

    # Create a BytesIO buffer to hold the PDF data
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Draw the content on the PDF
    p.drawString(100, height - 100, f'Report ID: {report.id}')
    p.drawString(100, height - 120, f'User: {report.user.username}')  # Adjust according to User model
    p.drawString(100, height - 140, f'Plant: {report.plant.name}')
    p.drawString(100, height - 160, f'Generated At: {report.generated_at.strftime("%Y-%m-%d %H:%M:%S")}')
    p.drawString(100, height - 180, 'Report Data:')
    text_object = p.beginText(100, height - 200)
    text_object.setFont("Helvetica", 12)

    # Add report data (if it's too long, consider wrapping it)
    for line in report.report_data.splitlines():
        text_object.textLine(line)
    
    p.drawText(text_object)

    # Finalize the PDF and get the value of the BytesIO buffer
    p.showPage()
    p.save()
    buffer.seek(0)

    # Create the HTTP response with the PDF
    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="report_{report_id}.pdf"'
    return response

    
def generate_report(request):
    # Implement report generation logic here
    return HttpResponse('New report generated')



from django.http import JsonResponse
from django.shortcuts import render
from .models import ImageUpload, Plant, Disease
from django.db.models import Count
from datetime import timedelta
from django.utils import timezone

def visualizations(request):
    return render(request, 'visualizations.html')

def visualization_data(request):
    # Get the current date
    now = timezone.now()
    
    # Data for Disease Occurrence
    disease_occurrences = (
        ImageUpload.objects
        .values('detected_disease__name')
        .annotate(count=Count('detected_disease'))
    )
    
    disease_names = [occ['detected_disease__name'] for occ in disease_occurrences]
    occurrence_counts = [occ['count'] for occ in disease_occurrences]

    # Data for Disease Severity
    severity_counts = (
        ImageUpload.objects
        .values('confidence_score')
        .annotate(count=Count('id'))
    )
    
    severity_levels = ['Low', 'Medium', 'High']
    severity_distribution = [0, 0, 0]  # Assuming low < 0.5, medium < 0.75, high >= 0.75

    for upload in severity_counts:
        if upload['confidence_score'] < 0.5:
            severity_distribution[0] += upload['count']  # Low
        elif upload['confidence_score'] < 0.75:
            severity_distribution[1] += upload['count']  # Medium
        else:
            severity_distribution[2] += upload['count']  # High

    # Data for Timeline of Detections
    timeline_data = (
        ImageUpload.objects
        .filter(uploaded_at__gte=now - timedelta(days=90))
        .values('uploaded_at__date')
        .annotate(count=Count('id'))
    )
    
    timeline_labels = [data['uploaded_at__date'].strftime('%Y-%m-%d') for data in timeline_data]
    timeline_counts = [data['count'] for data in timeline_data]

    # Data for Disease Spread Map
    disease_map_data = (
        ImageUpload.objects
        .values('plant__name', 'detected_disease__name', 'uploaded_at', 'user__username')
    )

    map_markers = [
        {
            'lat': 51.5,  # Replace with actual latitude
            'lng': -0.09,  # Replace with actual longitude
            'disease': upload['detected_disease__name'],
            'popup': f"{upload['user__username']} detected {upload['detected_disease__name']} on {upload['uploaded_at'].strftime('%Y-%m-%d')}"
        }
        for upload in disease_map_data
    ]

    response_data = {
        'disease_occurrences': {
            'labels': disease_names,
            'data': occurrence_counts
        },
        'severity_distribution': {
            'labels': severity_levels,
            'data': severity_distribution
        },
        'timeline': {
            'labels': timeline_labels,
            'data': timeline_counts
        },
        'map_markers': map_markers,
    }
    
    return JsonResponse(response_data)

# API view to get report data for charts
def get_report_data(request):
    # Fetch all reports, you can filter by user, date, or plant type if needed
    reports = Report.objects.all()

    # Structure the data for the charts
    disease_occurrences = {}
    severity_distribution = {'low': 0, 'medium': 0, 'high': 0}
    timeline_data = {}

    # Loop through reports and populate data
    for report in reports:
        report_data = json.loads(report.report_data)

        # Count occurrences for each disease
        disease = report_data.get('disease', 'Unknown')
        if disease in disease_occurrences:
            disease_occurrences[disease] += 1
        else:
            disease_occurrences[disease] = 1

        # Count severity
        severity = report_data.get('severity', 'low')
        severity_distribution[severity] += 1

        # Track timeline data (assuming report_data has a date field or use generated_at)
        report_month = report.generated_at.strftime('%b')
        if report_month in timeline_data:
            timeline_data[report_month] += 1
        else:
            timeline_data[report_month] = 1

    # Prepare data for charts
    data = {
        'disease_occurrences': disease_occurrences,
        'severity_distribution': severity_distribution,
        'timeline_data': timeline_data,
    }

    return JsonResponse(data)

# API view to get data for the map
def get_map_data(request):
    reports = Report.objects.all()
    map_data = []

    for report in reports:
        report_data = json.loads(report.report_data)
        lat = report_data.get('latitude')
        lon = report_data.get('longitude')
        disease = report_data.get('disease', 'Unknown')

        if lat and lon:
            map_data.append({
                'lat': lat,
                'lon': lon,
                'disease': disease,
                'severity': report_data.get('severity', 'low'),
            })

    return JsonResponse({'map_data': map_data})


def visualization_view(request):
    # Fetch data for visualizations
    disease_occurrences = {}
    severity_distribution = {'Low': 0, 'Medium': 0, 'High': 0}
    timeline_data = {}

    # Query ImageUpload objects with related data
    uploads = ImageUpload.objects.select_related('detected_disease').all()

    # Process each upload to gather necessary data
    for upload in uploads:
        disease_name = upload.detected_disease.name if upload.detected_disease else 'Unknown'
        
        # Count occurrences of each disease
        if disease_name in disease_occurrences:
            disease_occurrences[disease_name] += 1
        else:
            disease_occurrences[disease_name] = 1

        # Count severity distribution based on confidence score
        confidence_score = upload.confidence_score
        if confidence_score < 0.5:
            severity_distribution['Low'] += 1
        elif confidence_score < 0.8:
            severity_distribution['Medium'] += 1
        else:
            severity_distribution['High'] += 1

        # Prepare timeline data based on upload date
        upload_date = upload.uploaded_at.date()
        if upload_date in timeline_data:
            timeline_data[upload_date] += 1
        else:
            timeline_data[upload_date] = 1

    # Prepare the context for the template
    context = {
        'disease_occurrences': disease_occurrences,
        'severity_distribution': severity_distribution,
        'timeline_data': timeline_data,
    }

    return render(request, 'visualizations.html', context)


@login_required
def profile_management(request):
    if request.method == 'POST':
        # Update user details
        user = request.user
        user.username = request.POST.get('name', user.username)
        user.email = request.POST.get('email', user.email)
        password = request.POST.get('password')
        if password:
            user.set_password(password)
        user.save()
        return redirect('profile_management')  # Redirect after saving changes

    return render(request, 'profile_management.html')





import os
import google.generativeai as genai
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Plant, Disease, ImageUpload
from django.conf import settings
import json
import tempfile

genai.configure(api_key= "AIzaSyCG0gJTWz2-nM6njzaJujaY__B4olyRc6w")

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    return file

def detect_disease(image_path, plant_name):
    """Uses Gemini API to detect plant disease."""
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction="You are a plant disease detection system. User sends inputs name and image. You give the detected plant type, disease, description, recommendation, and confidence score. if image doesnt looks plant image display please upload pant image",
    )

    file = upload_to_gemini(image_path, mime_type="image/jpeg")

    chat_session = model.start_chat(
        history=[
            {"role": "user", "parts": [file, f"plant name: {plant_name}"]},
        ]
    )

    response = chat_session.send_message("Detect the disease and provide details.")
    return json.loads(response.text)  # Assuming the response is in JSON format


@login_required
def upload_image(request):
    if request.method == 'POST':
        plant_name = request.POST.get('plant_name')
        image_file = request.FILES.get('image')

        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            for chunk in image_file.chunks():
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        # Call your disease detection function (replace with actual logic)
        api_response = detect_disease(temp_file_path, plant_name)

        # Get or create plant instance
        plant, created = Plant.objects.get_or_create(name=plant_name, plant_type=api_response['plant_type'])

        # Get or create disease instance based on the detected disease
        disease, created = Disease.objects.get_or_create(
            name=api_response['disease'],
            description=api_response['description'],
            recommendations=api_response['recommendation']
        )

        # Save the uploaded image to ImageUpload model
        image_upload = ImageUpload.objects.create(
            user=request.user,  # Assuming the user is logged in
            plant=plant,
            image=image_file,
            detected_disease=disease,
            confidence_score=api_response['confidence_score'] * 100,  # Convert to percentage
            api_response=api_response
        )

        # Clean up the temporary file
        os.remove(temp_file_path)

        # Prepare the result to save in session
        result = {
            'plant_type': api_response['plant_type'],
            'disease': api_response['disease'],
            'description': api_response['description'],
            'recommendation': api_response['recommendation'],
            'confidence_score': api_response['confidence_score'] * 100  # Convert to percentage
        }

        # Store result in session for access on the results page
        request.session['result'] = result

        # Redirect to results page
        return redirect('detection_results')

    return render(request, 'upload_image.html')

@login_required
def detection_results(request):
    # Retrieve the result from the session
    result = request.session.get('result', None)
    return render(request, 'detection_results.html', {'result': result})

def visualizations(request):
    # Here you can pass dynamic data if needed in the future
    return render(request, 'visualizations.html')


from django.utils.timezone import now
def visualization(request):
    # Get all plants for the dropdown
    plants = Plant.objects.all()
    
    # Get filter criteria from request
    selected_plant = request.GET.get('plant', None)
    selected_severity = request.GET.get('severity', 'all')
    selected_days = int(request.GET.get('days', 7))
    
    # Calculate the date range
    time_threshold = now() - timedelta(days=selected_days)
    
    # Filter reports based on user selection
    reports = Report.objects.filter(generated_at__gte=time_threshold)
    
    if selected_plant:
        reports = reports.filter(plant_id=selected_plant)
    
    if selected_severity != 'all':
        # Make sure to filter by an actual field instead of 'report_data'
        reports = reports.filter(report_data__icontains=selected_severity)  # Adjust this if severity is stored elsewhere
    
    # Context for rendering in the template
    context = {
        'plants': plants,
        'reports': reports,
        'selected_plant': selected_plant,
        'selected_severity': selected_severity,
        'selected_days': selected_days,
    }
    
    return render(request, 'visualization.html', context)


import plotly.express as px
import plotly.graph_objects as go
from django.shortcuts import render
from django.db.models import Count, Avg
@login_required
def combined_visualization_view(request):
    user = request.user  # Get the authenticated user

    # --- 1. Plant Upload Count (Filter by user) ---
    plants = Plant.objects.filter(imageupload__user=user).annotate(upload_count=Count('imageupload')).order_by('-upload_count')
    plant_names = [plant.name for plant in plants]
    upload_counts = [plant.upload_count for plant in plants]

    fig1 = px.bar(
        x=plant_names,
        y=upload_counts,
        labels={'x': 'Plant', 'y': 'Number of Uploads'},
        title="Number of Image Uploads per Plant"
    )
    plant_upload_chart = fig1.to_html(full_html=False)

    # --- 2. Disease Count (Pie Chart, Filter by user) ---
    diseases = Disease.objects.filter(imageupload__user=user).annotate(detection_count=Count('imageupload')).order_by('-detection_count')
    disease_names = [disease.name for disease in diseases]
    detection_counts = [disease.detection_count for disease in diseases]

    fig2 = px.pie(
        names=disease_names,
        values=detection_counts,
        title="Most Commonly Detected Diseases",
    )
    disease_count_chart = fig2.to_html(full_html=False)

    # --- 3. Average Confidence Score (Filter by user) ---
    diseases_avg = Disease.objects.filter(imageupload__user=user).annotate(avg_confidence=Avg('imageupload__confidence_score')).filter(avg_confidence__gt=0.00).order_by('-avg_confidence')
    avg_disease_names = [disease.name for disease in diseases_avg]
    avg_confidences = [disease.avg_confidence for disease in diseases_avg]

    fig3 = px.bar(
        x=avg_disease_names,
        y=avg_confidences,
        labels={'x': 'Disease', 'y': 'Average Confidence Score'},
        title="Average Confidence Score of Disease Detection"
    )
    avg_confidence_chart = fig3.to_html(full_html=False)

    # --- 4. Disease Distribution by Plant Type (Filter by user) ---
    plant_disease_data = {}
    for plant in plants:
        disease_counts = ImageUpload.objects.filter(user=user, plant=plant, confidence_score__gt=0.00).values('detected_disease__name').annotate(Count('detected_disease')).order_by('-detected_disease__count')
        plant_disease_data[plant.name] = {entry['detected_disease__name']: entry['detected_disease__count'] for entry in disease_counts}

    # Pass all data to the template
    context = {
        'plant_upload_chart': plant_upload_chart,
        'disease_count_chart': disease_count_chart,
        'avg_confidence_chart': avg_confidence_chart,
        'plant_disease_data': plant_disease_data,
    }

    return render(request, 'combined_visualization.html', context)


import csv
from django.http import HttpResponse
from django.db.models import Avg, Count
import io
from django.core.files.base import ContentFile
from django.http import HttpResponse
from django.utils import timezone
@login_required
def download_report_view(request):
    user = request.user  # Get the authenticated user

    # Create an in-memory text stream to store the CSV content
    csv_buffer = io.StringIO()

    # Create a CSV writer object
    writer = csv.writer(csv_buffer)
    
    # Write the header of the CSV file
    writer.writerow(['Plant', 'Number of Uploads', 'Disease', 'Detection Count', 'Average Confidence Score'])
    
    # Fetch data from the database (Filter by user)
    plants = Plant.objects.filter(imageupload__user=user).annotate(upload_count=Count('imageupload'))
    for plant in plants:
        # Fetch diseases related to this plant and user
        plant_uploads = ImageUpload.objects.filter(user=user, plant=plant).select_related('detected_disease')
        diseases = plant_uploads.values('detected_disease__name').annotate(
            detection_count=Count('detected_disease'),
            avg_confidence=Avg('confidence_score')
        )

        for disease in diseases:
            writer.writerow([
                plant.name,
                plant.upload_count,
                disease['detected_disease__name'],
                disease['detection_count'],
                round(disease['avg_confidence'], 2) if disease['avg_confidence'] else 'N/A'
            ])

    # Get the CSV content from the buffer
    csv_content = csv_buffer.getvalue()
    csv_buffer.close()

    # Save the CSV content to the Report model
    if user.is_authenticated:
        report = Report(
            user=user,
            plant=plants.first() if plants.exists() else None,  # Assuming you want to link to the first plant
            generated_at=timezone.now(),
            report_data=csv_content
        )

        # Save the file to the 'report_file' field
        csv_filename = f"plant_disease_report_{timezone.now().strftime('%Y%m%d_%H%M%S')}.csv"
        report.report_file.save(csv_filename, ContentFile(csv_content.encode('utf-8')))
        report.save()

    # Return the CSV as a downloadable response
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{csv_filename}"'
    response.write(csv_content)
    
    return response
