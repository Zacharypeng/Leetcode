/*
    Keep updating the maxDis for the window below
    |cur -- curEnd|
    once hit the window end, make a jump
*/
class Solution {
    public int jump(int[] nums) {
        int jump = 0, maxDis = 0, cur = 0, curEnd = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            maxDis = Math.max(maxDis, i + nums[i]);
            if (i == curEnd) {
                jump++;
                curEnd = maxDis;
            }
        }
        return jump;
    }
}