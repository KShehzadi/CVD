from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from uploads.core.models import Document
from uploads.core.forms import DocumentForm
from collections import deque
import numpy as np
import argparse
import cv2
import time
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage



def home(request):
    documents = Document.objects.all()
    return render(request, 'core/home.html', { 'documents': documents })

def negative_filter(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        file = cv2.imread('C:/djangosample2/simple-file-upload/'+uploaded_file_url,1)
        negativefile = cv2.bitwise_not(file)
        cv2.imwrite("media/negativefile.png", negativefile)
        nuploaded_file_url = fs.url('negativefile.png')
        return render(request, 'core/negative_filter.html', {
            'uploaded_file_url': uploaded_file_url, 'nuploaded_file_url':nuploaded_file_url
        })
    return render(request, 'core/negative_filter.html')
def Linear_Averaging(request):
    if request.method == 'POST' and request.FILES['myfile']:
        if request.POST['Averaging']:
            averagingvalue = int(request.POST['Averaging'])
        else:
            averagingvalue = 1
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        file = cv2.imread('C:/djangosample2/simple-file-upload/'+uploaded_file_url,1)
        linear_averaging = cv2.blur(file,(averagingvalue,averagingvalue))
        cv2.imwrite("media/Linear_Averaging.png", linear_averaging)
        nuploaded_file_url = fs.url('Linear_Averaging.png')
        return render(request, 'core/Linear_Averaging.html', {
            'uploaded_file_url': uploaded_file_url, 'nuploaded_file_url':nuploaded_file_url
        })
    return render(request, 'core/Linear_Averaging.html')
def Linear_Laplacian(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        file = cv2.imread('C:/djangosample2/simple-file-upload/'+uploaded_file_url,1)
        linear_laplacian = cv2.Laplacian(file,cv2.CV_64F)
        cv2.imwrite("media/Linear_Laplacian.png", linear_laplacian*255)
        nuploaded_file_url = fs.url('Linear_Laplacian.png')
        return render(request, 'core/Linear_Laplacian.html', {
            'uploaded_file_url': uploaded_file_url, 'nuploaded_file_url':nuploaded_file_url
        })
    return render(request, 'core/Linear_Laplacian.html')

def Linear_Gaussian(request):
    if request.method == 'POST' and request.FILES['myfile']:
        if request.POST['GaussianValue']:
            kernelvalue = int(request.POST['GaussianValue'])
        else:
            kernelvalue = 5
        kernel = (kernelvalue,kernelvalue)
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        file = cv2.imread('C:/djangosample2/simple-file-upload/'+uploaded_file_url,1)
        linear_gaussian = cv2.GaussianBlur(file,kernel,0)
        cv2.imwrite("media/Linear_Gaussian.png", linear_gaussian)
        nuploaded_file_url = fs.url('Linear_Gaussian.png')
        return render(request, 'core/Linear_Gaussian.html', {
            'uploaded_file_url': uploaded_file_url, 'nuploaded_file_url':nuploaded_file_url
        })
    return render(request, 'core/Linear_Gaussian.html')


def histogram_match(request):
    if request.method == 'POST' and request.FILES['myfile'] and request.FILES['myfile1']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        myfile1 = request.FILES['myfile1']
        fs1 = FileSystemStorage()
        filename1 = fs1.save(myfile1.name, myfile1)
        uploaded_file_url1 = fs1.url(filename1)
        file = cv2.imread('C:/djangosample2/simple-file-upload/'+uploaded_file_url,0)
        file1 = cv2.imread('C:/djangosample2/simple-file-upload/'+uploaded_file_url1,0)
        a = hist_match(file, file1)
        cv2.imwrite("media/histogram_match.png", a)
        nuploaded_file_url = fs.url('histogram_match.png')
        return render(request, 'core/histogram_match.html', {
            'uploaded_file_url': uploaded_file_url,'uploaded_file_url1':uploaded_file_url1, 'nuploaded_file_url':nuploaded_file_url
        })
    return render(request, 'core/histogram_match.html')
def find_nearest_above(my_array, target):
    diff = my_array - target
    mask = np.ma.less_equal(diff, -1)
    # We need to mask the negative differences
    # since we are looking for values above
    if np.all(mask):
        c = np.abs(diff).argmin()
        return c # returns min index of the nearest if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()

def hist_match(original, specified):

    oldshape = original.shape
    original = original.ravel()
    specified = specified.ravel()

    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(original, return_inverse=True,return_counts=True)
    t_values, t_counts = np.unique(specified, return_counts=True)

    # Calculate s_k for original image
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    
    # Calculate s_k for specified image
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # Round the values
    sour = np.around(s_quantiles*255)
    temp = np.around(t_quantiles*255)
    
    # Map the rounded values
    b=[]
    for data in sour[:]:
        b.append(find_nearest_above(temp,data))
    b= np.array(b,dtype='uint8')

    return b[bin_idx].reshape(oldshape)

def histogram_equalization(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        file = cv2.imread('C:/djangosample2/simple-file-upload/'+uploaded_file_url,1)
        img_yuv = cv2.cvtColor(file, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # convert the YUV image back to RGB format
        equalizehist = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        cv2.imwrite("media/histogram_equalization.png", equalizehist)
        nuploaded_file_url = fs.url('histogram_equalization.png')
        return render(request, 'core/histogram_equalization.html', {
            'uploaded_file_url': uploaded_file_url, 'nuploaded_file_url':nuploaded_file_url
        })
    return render(request, 'core/histogram_equalization.html')

def gamma_correction_filter(request):
    if request.method == 'POST' and request.FILES['myfile']:
        if request.POST['gamma']:
            gammavalue = float(request.POST['gamma'])
        else:
            gammavalue = 1.0
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        file = cv2.imread('C:/djangosample2/simple-file-upload/'+uploaded_file_url,1)
        gammafile = adjust_gamma(file, gammavalue)
        cv2.imwrite("media/gamma_correction_filter.png", gammafile)
        nuploaded_file_url = fs.url('gamma_correction_filter.png')
        return render(request, 'core/gamma_correction_filter.html', {
            'uploaded_file_url': uploaded_file_url, 'nuploaded_file_url':nuploaded_file_url, 'gammavalue':gammavalue
        })
    return render(request, 'core/gamma_correction_filter.html')

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:

        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        file = cv2.imread('C:/djangosample2/simple-file-upload/'+uploaded_file_url,1)
        negativefile = cv2.bitwise_not(file)
        cv2.imwrite("media/negativefile.png", negativefile)
        nuploaded_file_url = fs.url('negativefile.png')
        return render(request, 'core/simple_upload.html', {
            'uploaded_file_url': uploaded_file_url, 'nuploaded_file_url':nuploaded_file_url
        })
    return render(request, 'core/simple_upload.html')

def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = DocumentForm()
    return render(request, 'core/model_form_upload.html', {
        'form': form
    })
