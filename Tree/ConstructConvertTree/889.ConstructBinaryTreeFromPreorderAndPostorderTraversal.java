/*
	1 (2, (4, 5) ) , (3, (6, 7) )
	( (4, 5), 2), ( (6, 7), 3), 1

	Build map on post order
	find index of the next root in post order
*/

class Solution {
	public TreeNode constructFromPrePost(int[] pre, int[] post) {
		Map<Integer, Integer> map = new HashMap<>();
		for (int i = 0; i < pre.length; i++) map.put(post[i], i);
		return helper(pre, 0, pre.length-1, post, 0, post.length-1, map);
	}

	private TreeNode helper(int[] pre, int preStart, int preEnd, int[] post, int postStart, int postEnd, Map<Integer, Integer> map) {
		if (preStart > preEnd || postStart > postEnd) return null;
		TreeNode root = new TreeNode(pre[preStart]);
		if (preStart + 1 <= preEnd) {
			int indexOfNextRoot = map.get(pre[preStart+1]);
			int numOfLeft = indexOfNextRoot - postStart + 1;
			root.left = helper(pre, preStart+1, preStart+numOfLeft, post, postStart, indexOfNextRoot, map);
			root.right = helper(pre, preStart+numOfLeft+1, preEnd, post, indexOfNextRoot+1, postEnd-1, map);
		}
		return root;
	}
}