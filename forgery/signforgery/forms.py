from django import forms
from signforgery.models import Image

class UpdateForm(forms.ModelForm):
    class Meta:
        model= Image
        fields= ["name", "imagefile"]
