/*
    Stack stores the index
    By once meet the bigger element and poping stack
    to maintain a monotonous increasing stack
*/
class Solution {
    public int[] nextGreaterElements(int[] nums) {
        int[] res = new int[nums.length];
        Arrays.fill(res, -1);
        Stack<Integer> stack = new Stack<>();
        int len = nums.length;
        for (int i = 0; i < len * 2; i++) {
            int cur = nums[i % len];
            while (!stack.isEmpty() && cur > nums[stack.peek()]) {
                res[stack.pop()] = cur;
            }
            stack.push(i % len);
        }
        return res;
    }
}