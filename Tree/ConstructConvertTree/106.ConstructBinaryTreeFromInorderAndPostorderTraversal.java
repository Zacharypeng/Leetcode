/*
	Use post order to find the root backward
	find the location of the root in in order array, left if left tree and right is right tree
	then find out the length of the left tree
	in the recursion, REMEMBER to exclude out the root in post order
	if ever get confused by the index, try to run a simple example
*/

class Solution {
	public TreeNode buildTree(int[] inorder, int[] postorder) {
		Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) map.put(inorder[i], i);
        return helper(postorder, 0, postorder.length-1, inorder, 0, inorder.length-1, map);
	}

	private TreeNode helper(int[] postorder, int postLeft, int postRight, int[] inorder, int inStart, int inEnd, Map<Integer, Integer> map) {
		if (postLeft > postRight || inStart > inEnd) return null;
		TreeNode root = new TreeNode(postorder[postRight]);
		int index = map.get(root.val);
		int numLeft = index - inStart;
		root.left = helper(postorder, postLeft, postLeft+numLeft-1, inorder, inStart, index-1, map);
		root.right = helper(postorder, postLeft+numLeft, postRight-1, inorder, index+1, inEnd, map);
		return root;
	}
}