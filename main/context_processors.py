import datetime

def add_current_year(request):
    return {
        'current_year': datetime.date.today().year
    }