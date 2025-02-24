from django.shortcuts import render, redirect, get_object_or_404
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from .models import Item, Category, Order
from django.contrib.auth import authenticate, login
from django.contrib import messages
from .forms import UserLoginForm, UserRegisterForm, ItemForm, CategoryForm, OrderForm
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse

def home(request):
    categories = Category.objects.all()
    featured_items = Item.objects.filter(is_featured=True)
    return render(request, 'catalog/index.html', {'categories': categories, 'featured_items': featured_items})

def about_us(request):
    return render(request, 'catalog/about_us.html')

def contact(request):
    return render(request, 'catalog/contact.html')

def login_view(request):
    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')  # Redirect to the home page after successful login
            else:
                messages.error(request, "Invalid username or password")
        else:
            messages.error(request, "Invalid form submission")
    else:
        form = UserLoginForm()
    return render(request, 'catalog/login.html', {'form': form})  # Corrected template path

def terms_and_conditions(request):
    return render(request, 'catalog/term_and_condition.html')

def register_view(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            if 'login_after_register' in request.POST:
                login(request, user)
                messages.success(request, f'Welcome {user.username}, you are now registered!')
                return redirect('home')
            else:
                messages.success(request, f'Account created successfully for {user.username}!')
                return redirect('login')
        else:
            messages.error(request, 'Registration failed. Please correct the errors below.')
    else:
        form = UserRegisterForm()
    return render(request, 'catalog/registration.html', {'form': form})  # Corrected template path

def item_list(request):
    items = Item.objects.all()
    return render(request, 'catalog/item_list.html', {'items': items})

def item_detail(request, pk):
    item = get_object_or_404(Item, pk=pk)
    return render(request, 'catalog/item_detail.html', {'item': item})

def item_create(request):
    if request.method == 'POST':
        form = ItemForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('item_list')
    else:
        form = ItemForm()
    return render(request, 'catalog/item_form.html', {'form': form})

def item_update(request, pk):
    item = get_object_or_404(Item, pk=pk)
    if request.method == 'POST':
        form = ItemForm(request.POST, request.FILES, instance=item)
        if form.is_valid():
            form.save()
            return redirect('item_list')
    else:
        form = ItemForm(instance=item)
    return render(request, 'catalog/item_form.html', {'form': form})

def item_delete(request, pk):
    item = get_object_or_404(Item, pk=pk)
    if request.method == 'POST':
        item.delete()
        return redirect('item_list')
    return render(request, 'catalog/item_confirm_delete.html', {'item': item})

def category_list(request):
    categories = Category.objects.all()
    return render(request, 'catalog/category_list.html', {'categories': categories})

def category_detail(request, pk):
    category = get_object_or_404(Category, pk=pk)
    return render(request, 'catalog/category_detail.html', {'category': category})

def category_create(request):
    if request.method == 'POST':
        form = CategoryForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('category_list')
    else:
        form = CategoryForm()
    return render(request, 'catalog/category_form.html', {'form': form})

def category_update(request, pk):
    category = get_object_or_404(Category, pk=pk)
    if request.method == 'POST':
        form = CategoryForm(request.POST, instance=category)
        if form.is_valid():
            form.save()
            return redirect('category_list')
    else:
        form = CategoryForm(instance=category)
    return render(request, 'catalog/category_form.html', {'form': form})

def category_delete(request, pk):
    category = get_object_or_404(Category, pk=pk)
    if request.method == 'POST':
        category.delete()
        return redirect('category_list')
    return render(request, 'catalog/category_confirm_delete.html', {'category': category})

@login_required
def create_order(request, item_id):
    item = get_object_or_404(Item, pk=item_id)
    if request.method == 'POST':
        form = OrderForm(request.POST)
        if form.is_valid():
            order = form.save(commit=False)
            order.user = request.user
            order.item = item
            order.total_price = item.price * order.quantity
            order.status = 'pending'  # Set status to pending
            order.save()
            print(f"Order saved: {order}")  # Debugging statement
            messages.success(request, 'Order placed successfully! Your order is pending approval.')
            return JsonResponse({'success': True})
        else:
            print("Form is not valid")  # Debugging statement
            print(form.errors)  # Debugging statement to print form errors
            messages.error(request, 'Please correct the errors below.')
            return JsonResponse({'success': False, 'errors': form.errors})
    else:
        form = OrderForm(initial={'item': item})
    return render(request, 'catalog/order_form.html', {'form': form, 'item': item})

@login_required
def order_location(request, order_id):
    order = get_object_or_404(Order, pk=order_id)
    if request.method == 'POST':
        location = request.POST.get('location')
        if location:
            order.location = location
            order.save()
            messages.success(request, 'Location saved successfully!')
            return redirect('catalog:order_history')
        else:
            messages.error(request, 'Please provide a location.')
    return render(request, 'catalog/order_location.html', {'order': order})

@login_required
def order_history(request):
    orders = Order.objects.filter(user=request.user)
    return render(request, 'catalog/order_history.html', {'orders': orders})

def test_view(request):
    return HttpResponse("URL pattern is working correctly.")

