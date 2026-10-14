import 'package:flutter/material.dart';
import '../services/api_service.dart';

class PendingApprovalsScreen extends StatefulWidget {
  const PendingApprovalsScreen({super.key});
  @override
  State<PendingApprovalsScreen> createState() => _PendingApprovalsScreenState();
}

class _PendingApprovalsScreenState extends State<PendingApprovalsScreen> {
  late Future<List<dynamic>> _future;

  @override
  void initState() {
    super.initState();
    _reload();
  }

  void _reload() { _future = ApiService.getPending(); }

  Future<void> _approve(int id) async {
    final res = await ApiService.approve(id);
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(res['message'] ?? 'Approved')));
    setState(_reload);
  }

  Future<void> _reject(int id) async {
    final ctrl = TextEditingController();
    final reason = await showDialog<String>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Dahilan ng Reject'),
        content: TextField(
          controller: ctrl,
          decoration: const InputDecoration(hintText: 'Optional...'),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(_), child: const Text('Kansel')),
          ElevatedButton(
            onPressed: () => Navigator.pop(_, ctrl.text),
            child: const Text('Reject')),
        ],
      ),
    );
    if (reason == null) return;
    final res = await ApiService.reject(id, reason: reason);
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(res['message'] ?? 'Rejected')));
    setState(_reload);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Pending Approvals'),
        backgroundColor: Colors.orange, foregroundColor: Colors.white,
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
            return const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.check_circle, size: 64, color: Colors.green),
                  SizedBox(height: 12),
                  Text('Walang pending. All clear!'),
                ],
              ),
            );
          }
          return ListView.builder(
            padding: const EdgeInsets.all(8),
            itemCount: list.length,
            itemBuilder: (c, i) {
              final a = list[i];
              final author = a['user']?['name'] ?? 'Unknown';
              final role = a['user']?['role'] ?? '';
              return Card(
                margin: const EdgeInsets.symmetric(vertical: 6),
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(a['title'] ?? '',
                        style: const TextStyle(
                          fontWeight: FontWeight.bold, fontSize: 16)),
                      const SizedBox(height: 4),
                      Text('Ni: $author ($role)',
                        style: const TextStyle(fontSize: 12, color: Colors.grey)),
                      const Divider(),
                      Text(a['content'] ?? ''),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          Expanded(
                            child: ElevatedButton.icon(
                              onPressed: () => _approve(a['id']),
                              icon: const Icon(Icons.check, size: 18),
                              label: const Text('APPROVE'),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.green,
                                foregroundColor: Colors.white),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: ElevatedButton.icon(
                              onPressed: () => _reject(a['id']),
                              icon: const Icon(Icons.close, size: 18),
                              label: const Text('REJECT'),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.red,
                                foregroundColor: Colors.white),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              );
            },
          );
        },
      ),
    );
  }
}
