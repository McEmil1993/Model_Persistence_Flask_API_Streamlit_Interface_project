import 'package:flutter/material.dart';
import '../services/api_service.dart';
import 'officers_screen.dart';
import 'announcements_screen.dart';
import 'create_post_screen.dart';
import 'pending_approvals_screen.dart';
import 'my_posts_screen.dart';
import 'login_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String? _role;
  String? _name;
  bool _canApprove = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final r = await ApiService.getRole();
    final n = await ApiService.getName();
    setState(() {
      _role = r;
      _name = n;
      _canApprove = ['president', 'vice_president', 'admin'].contains(r);
    });
  }

  String _roleLabel(String? r) {
    switch (r) {
      case 'president': return 'President';
      case 'vice_president': return 'Vice President';
      case 'officer': return 'Officer';
      case 'multimedia': return 'Multimedia Artist';
      case 'admin': return 'Admin';
      default: return 'Student';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('TMC Mabini Student Alliance'),
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () async {
              await ApiService.logout();
              if (!mounted) return;
              Navigator.pushAndRemoveUntil(context,
                MaterialPageRoute(builder: (_) => const LoginScreen()),
                (_) => false);
            },
          ),
        ],
      ),
      body: Column(
        children: [
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            color: Colors.blue.shade50,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Kumusta, ${_name ?? ''}!',
                  style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                Text('Role: ${_roleLabel(_role)}',
                  style: TextStyle(color: Colors.blue.shade700)),
              ],
            ),
          ),
          Expanded(
            child: GridView.count(
              padding: const EdgeInsets.all(16),
              crossAxisCount: 2,
              crossAxisSpacing: 12,
              mainAxisSpacing: 12,
              children: [
                _card(context, Icons.campaign, 'Announcements',
                  const AnnouncementsScreen(), Colors.blue),
                _card(context, Icons.people, 'Officers',
                  const OfficersScreen(), Colors.indigo),
                _card(context, Icons.image, 'Multimedia',
                  const OfficersScreen(department: 'Multimedia'), Colors.purple),
                _card(context, Icons.edit_note, 'Mag-post',
                  const CreatePostScreen(), Colors.teal),
                if (_canApprove)
                  _card(context, Icons.approval, 'Pending Approvals',
                    const PendingApprovalsScreen(), Colors.orange),
                _card(context, Icons.history, 'Aking mga Post',
                  const MyPostsScreen(), Colors.green),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _card(BuildContext c, IconData icon, String label, Widget page, Color color) {
    return InkWell(
      onTap: () => Navigator.push(c, MaterialPageRoute(builder: (_) => page)),
      child: Card(
        elevation: 3,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 48, color: color),
            const SizedBox(height: 8),
            Text(label,
              textAlign: TextAlign.center,
              style: const TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
      ),
    );
  }
}
