import 'package:flutter/material.dart';
import 'screens/login_screen.dart';

void main() => runApp(const TmcApp());

class TmcApp extends StatelessWidget {
  const TmcApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'TMC Mabini Student Alliance',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.blue, useMaterial3: true),
      home: const LoginScreen(),
    );
  }
}
