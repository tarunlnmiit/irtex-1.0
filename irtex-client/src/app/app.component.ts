import { Component, OnInit } from '@angular/core';
import { AppService } from './app.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'irtex-client';
  userdata;
  constructor(private appService: AppService) {}
  ngOnInit() {
    this.userdata = this.appService.getUserData();
  }
}
