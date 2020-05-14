import { Injectable } from '@angular/core';
import { HttpClient} from '@angular/common/http';

const localUrl = 'http://127.0.0.1:8000/userdata/';
// const localUrl = 'assets/data/smartphone.json';

@Injectable({
  providedIn: 'root'
})
export class AppService {
  constructor(private http: HttpClient) { }
  getUserData() {
    return this.http.get(localUrl);
  }
}
