/*
	Greedly push forward the boudary
	Notice:
		in the loop, i should be less than max
*/
class Solution {
    public boolean canJump(int[] nums) {
        int max = 0;
        for (int i = 0; i <= max; i++) {
            max = Math.max(max, i + nums[i]);
            if (max >= nums.length - 1) 
                return true;
        }
        return false;
    }
}