<?php
namespace App\Http\Controllers\Api;
use App\Http\Controllers\Controller;
use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;

class AuthController extends Controller {
    public function register(Request $r) {
        $data = $r->validate([
            'name' => 'required',
            'email' => 'required|email|unique:users',
            'password' => 'required|min:6',
            'role' => 'required|in:student,officer,multimedia,president,vice_president,admin',
        ]);
        $data['password'] = Hash::make($data['password']);
        $user = User::create($data);
        return response()->json(['token' => $user->createToken('api')->plainTextToken, 'user' => $user]);
    }
    public function login(Request $r) {
        $r->validate(['email'=>'required|email','password'=>'required']);
        $user = User::where('email', $r->email)->first();
        if (!$user || !Hash::check($r->password, $user->password)) {
            return response()->json(['message'=>'Mali ang email o password.'], 401);
        }
        return response()->json(['token' => $user->createToken('api')->plainTextToken, 'user' => $user]);
    }
    public function logout(Request $r) {
        $r->user()->currentAccessToken()->delete();
        return response()->json(['message' => 'Logged out']);
    }
}
