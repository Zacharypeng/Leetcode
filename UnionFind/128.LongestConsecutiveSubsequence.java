/*
	Basic Union Find
	find neighbors(consecutive numbers)
	Notice! when there are dupilcated numbers!!
*/

class Solution {

	class UnionFind {
		int[] parents;

		UnionFind(int size) {
			this.parents = new int[size];
			for (int i = 0; i < size; i++) {
				parents[i] = i;
			}
		}

		public void union(int child1, int child2) {
			int parent1 = find(child1);
			int parent2 = find(child2);
			if (parent1 == parent2) return;
			parents[parent1] = parent2;
		}

		public int find(int child) {
			if (child != parents[child]) {
				parents[child] = find(parents[child]);
			}
			return parents[child];
		}

		public int maxRank() {
			int[] rank = new int[parents.length];			
			int max = 0;
			for (int i = 0; i < rank.length; i++) {
				int parent = find(i);
				rank[parent]++;
				max = Math.max(max, rank[parent]);
			}
			return max;
		}
	}

    public int longestConsecutive(int[] nums) {
    	Map<Integer, Integer> map = new HashMap<>();
    	UnionFind unionFind = new UnionFind(nums.length);
    	for (int i = 0; i < nums.length; i++) {
    		if (map.containsKey(nums[i])) continue;
    		map.put(nums[i], i);
    		if (map.containsKey(nums[i]-1)) {
    			unionFind.union(i, map.get(nums[i]-1));
    		}
    		if (map.containsKey(nums[i]+1)) {
    			unionFind.union(i, map.get(nums[i]+1));	
    		}
    	}
    	return unionFind.maxRank();
    }
}