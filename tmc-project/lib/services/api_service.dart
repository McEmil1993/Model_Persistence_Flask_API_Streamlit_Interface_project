import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class ApiService {
  // Android emulator: http://10.0.2.2:8000/api
  // iOS simulator:    http://127.0.0.1:8000/api
  // Physical phone:   http://IP_NG_PC:8000/api
  static const String baseUrl = 'http://10.0.2.2:8000/api';

  static Future<String?> getToken() async {
    final p = await SharedPreferences.getInstance();
    return p.getString('token');
  }

  static Future<Map<String, String>> _headers({bool auth = true}) async {
    final h = {'Accept': 'application/json', 'Content-Type': 'application/json'};
    if (auth) {
      final t = await getToken();
      if (t != null) h['Authorization'] = 'Bearer $t';
    }
    return h;
  }

  static Future<Map<String, dynamic>> login(String email, String password) async {
    final res = await http.post(
      Uri.parse('$baseUrl/login'),
      headers: await _headers(auth: false),
      body: jsonEncode({'email': email, 'password': password}),
    );
    final data = jsonDecode(res.body);
    if (res.statusCode == 200) {
      final p = await SharedPreferences.getInstance();
      await p.setString('token', data['token']);
      await p.setString('role', data['user']['role']);
      await p.setString('name', data['user']['name']);
    }
    return data;
  }

  static Future<String?> getRole() async {
    final p = await SharedPreferences.getInstance();
    return p.getString('role');
  }

  static Future<String?> getName() async {
    final p = await SharedPreferences.getInstance();
    return p.getString('name');
  }

  static Future<void> logout() async {
    try { await http.post(Uri.parse('$baseUrl/logout'), headers: await _headers()); } catch (_) {}
    final p = await SharedPreferences.getInstance();
    await p.clear();
  }

  static Future<List<dynamic>> getOfficers({String? department}) async {
    final url = department != null
        ? '$baseUrl/officers?department=$department'
        : '$baseUrl/officers';
    final res = await http.get(Uri.parse(url), headers: await _headers(auth: false));
    return jsonDecode(res.body);
  }

  static Future<List<dynamic>> getAnnouncements() async {
    final res = await http.get(Uri.parse('$baseUrl/announcements'),
        headers: await _headers(auth: false));
    return jsonDecode(res.body);
  }

  static Future<Map<String, dynamic>> createAnnouncement(String title, String content) async {
    final res = await http.post(
      Uri.parse('$baseUrl/announcements'),
      headers: await _headers(),
      body: jsonEncode({'title': title, 'content': content}),
    );
    return jsonDecode(res.body);
  }

  static Future<List<dynamic>> getMyAnnouncements() async {
    final res = await http.get(Uri.parse('$baseUrl/my-announcements'),
        headers: await _headers());
    return jsonDecode(res.body);
  }

  static Future<List<dynamic>> getPending() async {
    final res = await http.get(Uri.parse('$baseUrl/announcements-pending'),
        headers: await _headers());
    return jsonDecode(res.body);
  }

  static Future<Map<String, dynamic>> approve(int id) async {
    final res = await http.post(
        Uri.parse('$baseUrl/announcements/$id/approve'),
        headers: await _headers());
    return jsonDecode(res.body);
  }

  static Future<Map<String, dynamic>> reject(int id, {String? reason}) async {
    final res = await http.post(
        Uri.parse('$baseUrl/announcements/$id/reject'),
        headers: await _headers(),
        body: jsonEncode({'reason': reason ?? ''}));
    return jsonDecode(res.body);
  }
}
