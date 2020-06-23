# Generated by Django 2.2.12 on 2020-06-07 16:11

from django.db import migrations, models
import djongo.models.fields


class Migration(migrations.Migration):

    dependencies = [
        ('upload', '0004_queryimagesimilarity'),
    ]

    operations = [
        migrations.CreateModel(
            name='Session',
            fields=[
                ('_id', djongo.models.fields.ObjectIdField(auto_created=True, primary_key=True, serialize=False)),
                ('clicks', djongo.models.fields.DictField(null=True)),
                ('created', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]