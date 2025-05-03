from django.db import models

class DummyModel(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name


# The DummyModel in models.py is just a placeholder. Its Included bcoz Django expects the models.py file to exist even if you don’t use database models.
# This  System does not store anything in the database (like user profiles, ratings, or movie info).
#  Everything is handled in-memory using Pandas and the MovieLens dataset.
