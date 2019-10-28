/*
	find the root in the preorder
	then find the root in the inorder, everything in left is left tree, everything in right is right tree
	when recursion, REMEMBER to exclude ROOT!!! both preorder and inorder.
*/
class Solution {
	public TreeNode buildTree(int[] preorder, int[] inorder) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) map.put(inorder[i], i);        
        return helper(0, preorder.length-1, 0, inorder.length-1, preorder, inorder, map);
    }

    private TreeNode helper(int preStart, int preEnd, int inStart, int inEnd, int[] preorder, int[] inorder, Map<Integer, Integer> map) {
    	if (preStart > preEnd || inStart > inEnd) return null;
    	TreeNode root = new TreeNode(preorder[preStart]);
    	int indexOfRoot = map.get(root.val);
    	int numOfLeft = indexOfRoot - inStart;
    	root.left = helper(preStart+1, preStart+numOfLeft, inStart, indexOfRoot, preorder, inorder, map);
    	root.right = helper(preStart+numOfLeft+1, preEnd, indexOfRoot+1, inEnd, preorder, inorder, map);
    	return root;
    }
}