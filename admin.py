from django.contrib import admin
from .models import Item, Category, Order

class ItemAdmin(admin.ModelAdmin):
    list_display = ('name', 'price', 'category', 'is_for_rent', 'is_featured')
    search_fields = ('name', 'description')
    list_filter = ('category', 'is_for_rent', 'is_featured')

class CategoryAdmin(admin.ModelAdmin):
    list_display = ('name',)
    search_fields = ('name',)

@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'item', 'quantity', 'status', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('user__username', 'item__name')
    actions = ['approve_orders']

    def approve_orders(self, request, queryset):
        queryset.update(status='approved')
    approve_orders.short_description = 'Approve selected orders'

admin.site.register(Item, ItemAdmin)
admin.site.register(Category, CategoryAdmin)

admin.site.site_title = "Hike Gear Nepal Admin Portal"
admin.site.index_title = "Welcome to Hike Gear Nepal Admin Portal"
