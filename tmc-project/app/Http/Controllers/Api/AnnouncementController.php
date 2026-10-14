<?php
namespace App\Http\Controllers\Api;
use App\Http\Controllers\Controller;
use App\Models\Announcement;
use Illuminate\Http\Request;

class AnnouncementController extends Controller {

    public function index() {
        return Announcement::with(['user','officer'])
            ->where('status', 'approved')
            ->latest()->get();
    }

    public function show(Announcement $announcement) {
        return $announcement->load(['user','officer','approver']);
    }

    public function store(Request $r) {
        $user = $r->user();
        if (!$user->canPost()) {
            return response()->json(['message' => 'Hindi ka pwede mag-post.'], 403);
        }
        $data = $r->validate([
            'title'      => 'required|string|max:255',
            'content'    => 'required|string',
            'image'      => 'nullable|string',
            'officer_id' => 'nullable|exists:officers,id',
        ]);
        $autoApprove = $user->canApprove();
        $announcement = Announcement::create([
            'title' => $data['title'],
            'content' => $data['content'],
            'image' => $data['image'] ?? null,
            'officer_id' => $data['officer_id'] ?? null,
            'user_id' => $user->id,
            'status' => $autoApprove ? 'approved' : 'pending',
            'approved_by' => $autoApprove ? $user->id : null,
            'approved_at' => $autoApprove ? now() : null,
        ]);
        return response()->json([
            'message' => $autoApprove
                ? 'Na-post na agad (auto-approved).'
                : 'Na-submit. Naghihintay ng approval ng President o Vice President.',
            'data' => $announcement->load('user'),
        ], 201);
    }

    public function pending(Request $r) {
        if (!$r->user()->canApprove()) {
            return response()->json(['message' => 'Unauthorized.'], 403);
        }
        return Announcement::with(['user','officer'])
            ->where('status', 'pending')
            ->oldest()->get();
    }

    public function mine(Request $r) {
        return Announcement::with(['user','approver'])
            ->where('user_id', $r->user()->id)
            ->latest()->get();
    }

    public function approve(Request $r, Announcement $announcement) {
        if (!$r->user()->canApprove()) {
            return response()->json(['message' => 'Unauthorized.'], 403);
        }
        if ($announcement->status === 'approved') {
            return response()->json(['message' => 'Approved na ito.'], 400);
        }
        $announcement->update([
            'status' => 'approved',
            'approved_by' => $r->user()->id,
            'approved_at' => now(),
            'rejection_reason' => null,
        ]);
        return response()->json([
            'message' => 'Na-approve na.',
            'data' => $announcement->fresh(['user','approver']),
        ]);
    }

    public function reject(Request $r, Announcement $announcement) {
        if (!$r->user()->canApprove()) {
            return response()->json(['message' => 'Unauthorized.'], 403);
        }
        $data = $r->validate(['reason' => 'nullable|string']);
        $announcement->update([
            'status' => 'rejected',
            'approved_by' => $r->user()->id,
            'approved_at' => now(),
            'rejection_reason' => $data['reason'] ?? 'Walang dahilan.',
        ]);
        return response()->json([
            'message' => 'Na-reject.',
            'data' => $announcement->fresh(['user','approver']),
        ]);
    }

    public function destroy(Request $r, Announcement $announcement) {
        $user = $r->user();
        if ($announcement->user_id !== $user->id && !$user->canApprove()) {
            return response()->json(['message' => 'Unauthorized.'], 403);
        }
        $announcement->delete();
        return response()->json(['message' => 'Deleted']);
    }
}
