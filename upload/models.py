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
