from djongo import models


class UserData(models.Model):
    _id = models.ObjectIdField()
    created = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=64, null=False, default='Default Name')
    dob = models.DateField(null=False, default='1970-01-01')
    address = models.CharField(max_length=128, null=True)

    class Meta:
        ordering = ['created']
