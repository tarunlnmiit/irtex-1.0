# Generated by Django 2.2.12 on 2020-04-26 20:20

from django.db import migrations, models
import djongo.models.fields


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='UserData',
            fields=[
                ('_id', djongo.models.fields.ObjectIdField(auto_created=True, primary_key=True, serialize=False)),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('name', models.CharField(default='Default Name', max_length=64)),
                ('dob', models.DateField(default='1970-01-01')),
                ('address', models.CharField(max_length=128, null=True)),
            ],
            options={
                'ordering': ['created'],
            },
        ),
    ]
