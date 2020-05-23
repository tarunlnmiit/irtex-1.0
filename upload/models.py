from djongo import models


class QueryImage(models.Model):
    _id = models.ObjectIdField()
    file = models.FileField(blank=False, null=False)
    url = models.URLField(blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file.name

    class Meta:
        ordering = ['created']


class DatasetImage(models.Model):
    _id = models.ObjectIdField()
    name = models.CharField(max_length=128, null=False, blank=False)
    url = models.URLField(blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True)
    label = models.CharField(max_length=32, null=False, blank=False)
    metadata = models.DictField(null=True)


class QueryImageSimilarity(models.Model):
    _id = models.ObjectIdField()
    query_image_id = models.CharField(max_length=128, null=False, blank=False)
    similarities = models.ListField()
    created = models.DateTimeField(auto_now_add=True)