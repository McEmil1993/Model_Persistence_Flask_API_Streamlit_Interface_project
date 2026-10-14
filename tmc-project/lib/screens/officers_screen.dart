import 'package:flutter/material.dart';
import '../services/api_service.dart';

class OfficersScreen extends StatefulWidget {
  final String? department;
  const OfficersScreen({super.key, this.department});
  @override
  State<OfficersScreen> createState() => _OfficersScreenState();
}

class _OfficersScreenState extends State<OfficersScreen> {
  late Future<List<dynamic>> _future;

  @override
  void initState() {
    super.initState();
    _future = ApiService.getOfficers(department: widget.department);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.department ?? 'Officers'),
        backgroundColor: Colors.blue, foregroundColor: Colors.white,
      ),
      body: FutureBuilder(
        future: _future,
        builder: (c, snap) {
          if (snap.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snap.hasError) return Center(child: Text('Error: ${snap.error}'));
          final list = snap.data ?? [];
          if (list.isEmpty) return const Center(child: Text('No officers found'));
          return ListView.builder(
            padding: const EdgeInsets.all(8),
            itemCount: list.length,
            itemBuilder: (c, i) {
              final o = list[i];
              return Card(
                child: ListTile(
                  leading: CircleAvatar(
                    backgroundColor: Colors.blue.shade100,
                    child: Text(o['full_name'][0]),
                  ),
                  title: Text(o['full_name']),
                  subtitle: Text('${o['position']} • ${o['department'] ?? ''}'),
                ),
              );
            },
          );
        },
      ),
    );
  }
}
