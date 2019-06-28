from django.conf.urls import url
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from uploads.core import views
urlpatterns = [
    url(r'^$', views.home, name='home'),
    url(r'^uploads/simple/$', views.simple_upload, name='simple_upload'),
    url(r'^uploads/form/$', views.model_form_upload, name='model_form_upload'),
    url(r'^uploads/negative/$', views.negative_filter, name='negative_filter'),
    url(r'^uploads/gamma_correction_filter/$', views.gamma_correction_filter, name='gamma_correction_filter'),
    url(r'^uploads/histogram_equalization/$', views.histogram_equalization, name='histogram_equalization'),
    url(r'^uploads/histogram_match/$', views.histogram_match, name='histogram_match'),
    url(r'^uploads/Linear_Averaging/$', views.Linear_Averaging, name='Linear_Averaging'),
    url(r'^uploads/Linear_Gaussian/$', views.Linear_Gaussian, name='Linear_Gaussian'),
    url(r'^uploads/Linear_Laplacian/$', views.Linear_Laplacian, name='Linear_Laplacian'),
    url(r'^admin/', admin.site.urls),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
