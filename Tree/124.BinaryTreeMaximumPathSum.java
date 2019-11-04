/*
    initial value of max in main function is important
    max is to record any path sum that is not connected to the root
    helper function return the path sum that contains the root
*/
class Solution {
    int max;
    public int maxPathSum(TreeNode root) {
        max = root.val;
        return Math.max(helper(root), max);
    }
    
    private int helper(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null && root.right == null) {
            max = Math.max(max, root.val);
            return root.val;    
        }            
        int leftSum = Math.max(helper(root.left), 0);
        int rightSum = Math.max(helper(root.right), 0);
        max = Math.max(leftSum + rightSum + root.val, max);
        return Math.max(leftSum, rightSum) + root.val;
    }
}