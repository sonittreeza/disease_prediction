from django.db import models
from django.contrib.auth.models import User


# Create your models here.
class Plant(models.Model):
    name = models.CharField(max_length=100)
    plant_type = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
    

class Disease(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    recommendations = models.TextField()

    def __str__(self):
        return self.name
    

class ImageUpload(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    plant = models.ForeignKey(Plant, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='plant_images/')
    detected_disease = models.ForeignKey(Disease, on_delete=models.SET_NULL, null=True, blank=True)
    confidence_score = models.DecimalField(max_digits=5, decimal_places=2)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    api_response = models.JSONField(null=True, blank=True) 

    def __str__(self):
        return f'{self.user} - {self.plant}'


class Report(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    plant = models.ForeignKey('Plant', on_delete=models.CASCADE)
    generated_at = models.DateTimeField(auto_now_add=True)
    report_data = models.TextField()
    report_file = models.FileField(upload_to='reports/', null=True, blank=True)  # Field for saving report files

    def __str__(self):
        return f'Report {self.id} for {self.plant}'
    



   

