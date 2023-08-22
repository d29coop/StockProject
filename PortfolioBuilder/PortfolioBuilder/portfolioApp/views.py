from django.shortcuts import render
from django.http import HttpResponse
from . import portfolio_builder as pb

# Create your views here.
def hello(request):
    return render(request, 'hello.html', {'name':'Shankar'})
def home(request):
    return render(request, 'index.html')
def input(request):
    return render(request, 'input.html')

def output(request):
    if request.method == 'POST':
        years = float(request.POST.get('years'))
        amount = float(request.POST.get('amount'))
        risk_level = float(request.POST.get('risk_level'))
        print(years, amount, risk_level)
        print(pb.getOptPortfolio(amount, years, risk_level))
        
    return render(request, 'output.html')