<?php
namespace App\Http\Controllers\Api;
use App\Http\Controllers\Controller;
use App\Models\Officer;
use Illuminate\Http\Request;

class OfficerController extends Controller {
    public function index(Request $r) {
        $q = Officer::query();
        if ($r->department) $q->where('department', $r->department);
        return $q->orderBy('order_no')->get();
    }
    public function store(Request $r) {
        $data = $r->validate([
            'full_name'=>'required','position'=>'required','department'=>'nullable',
            'photo'=>'nullable','email'=>'nullable','contact'=>'nullable',
            'bio'=>'nullable','order_no'=>'integer',
        ]);
        return Officer::create($data);
    }
    public function show(Officer $officer) { return $officer; }
    public function update(Request $r, Officer $officer) {
        $officer->update($r->all());
        return $officer;
    }
    public function destroy(Officer $officer) {
        $officer->delete();
        return response()->json(['message'=>'Deleted']);
    }
}
