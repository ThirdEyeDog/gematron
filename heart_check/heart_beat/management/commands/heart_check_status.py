from django.core.management.base import BaseCommand
import requests
import time

class Command(BaseCommand):
    help = 'Checks the server status every 30 seconds'

    def handle(self, *args, **kwargs):
        while True:
            try:
                response = requests.get('http://127.0.0.1:8000/heart_beat/')
                if response.status_code == 200:
                    self.stdout.write("HEART IN GOOD CONDITION")
                else:
                    self.stdout.write("HEART NEED REPAIRS")
            except:
                self.stdout.write("HEART NEED REPAIRS")
            time.sleep(30)
