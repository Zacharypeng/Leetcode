/*
    Sort the Array! Because we want same number next to each other.
    Avoid duplicates! If the last number has visited, it means that all the permutations after last number
    has already added. So if the current number is equal to the last number and last number is visited,
    we should skip current number.

*/


class Solution {
    List<List<Integer>> ans = new ArrayList<>();
    public List<List<Integer>> permuteUnique(int[] nums) {          
        Arrays.sort(nums);
        helper(new ArrayList<Integer>(), nums, new boolean[nums.length]);        
        return ans;
    }
    
    private void helper(ArrayList<Integer> subList, int[] nums, boolean[] visited) {
        if (subList.size() == nums.length) {
            ans.add(new ArrayList<>(subList));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (visited[i] || ( i > 0 && !visited[i-1] && nums[i] == nums[i-1])) continue;
                subList.add(nums[i]);
                visited[i] = true;
                helper(subList, nums, visited);
                subList.remove(subList.size() - 1);
                visited[i] =false;
            
        }
    }
}