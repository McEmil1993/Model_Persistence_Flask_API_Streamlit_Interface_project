import 'package:flutter/material.dart';
import '../services/api_service.dart';

class MyPostsScreen extends StatefulWidget {
  const MyPostsScreen({super.key});
  @override
  State<MyPostsScreen> createState() => _MyPostsScreenState();
}

class _MyPostsScreenState extends State<MyPostsScreen> {
  late Future<List<dynamic>> _future;

  @override
  void initState() {
    super.initState();
    _future = ApiService.getMyAnnouncements();
  }

  Color _statusColor(String s) {
    switch (s) {
      case 'approved': return Colors.green;
      case 'rejected': return Colors.red;
      default: return Colors.orange;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Aking mga Post'),
        backgroundColor: Colors.green, foregroundColor: Colors.white,
      ),
      body: FutureBuilder(
        future: _future,
        builder: (c, snap) {
          if (snap.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snap.hasError) return Center(child: Text('Error: ${snap.error}'));
          final list = snap.data ?? [];
          if (list.isEmpty) {
            return const Center(child: Text('Wala ka pang na-post.'));
          }
          return ListView.builder(
            padding: const EdgeInsets.all(8),
            itemCount: list.length,
            itemBuilder: (c, i) {
              final a = list[i];
              final status = (a['status'] ?? 'pending').toString();
              return Card(
                child: ListTile(
                  leading: Icon(Icons.article, color: _statusColor(status)),
                  title: Text(a['title'] ?? ''),
                  subtitle: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(a['content'] ?? '',
                        maxLines: 2, overflow: TextOverflow.ellipsis),
                      const SizedBox(height: 4),
                      Chip(
                        label: Text(status.toUpperCase(),
                          style: const TextStyle(color: Colors.white, fontSize: 11)),
                        backgroundColor: _statusColor(status),
                        padding: EdgeInsets.zero,
                        materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                      ),
                      if (status == 'rejected' && a['rejection_reason'] != null)
                        Text('Dahilan: ${a['rejection_reason']}',
                          style: const TextStyle(fontSize: 11, color: Colors.red)),
                    ],
                  ),
                  isThreeLine: true,
                ),
              );
            },
          );
        },
      ),
    );
  }
}
