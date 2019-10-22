/*
	dp[i][j] means whether sum j can be formed from the first ith element in nums
	here, wo can NOT reuse element
	transformation fuction: dp[i][j] = dp[i-1][j] || dp[i-1][j-num]

	Notice: 
		one more space for the base cases!
		the different index for nums and dp!!!

	For the optimized space solution, loop from right to left
*/
class Solution {
	public boolean canPartition(int[] nums) {
		int sum = 0;
		for (int num : nums) sum += num;
		if (sum % 2 == 1) return false;
		sum /= 2;
		boolean[][] dp = new boolean[nums.length+1][sum+1];
		// set up the base cases
		for (int i = 0; i <= nums.length; i++) dp[i][0] = true;
		for (int i = 1; i <= nums.length; i++) {
			for (int j = 1; j <= sum; j++) {
				dp[i][j] = dp[i-1][j] || (j >= nums[i-1] ? dp[i-1][j-nums[i-1]] : false);
			}
		}
		return dp[nums.length][sum];
	}
}

// optimize space

class Solution {
	public boolean canPartition(int[] nums) {
		int sum = 0;
		for (int num : nums) sum += num;
		if (sum % 2 == 1) return false;
		sum /= 2;
		boolean[] dp = new boolean[sum+1];
		dp[0] = true;
		for (int num: nums) {
			for (int i = sum; i > 0; i--) {
				dp[i] |= (i >= num ? dp[i-num] : false) ;
			}
		}
		return dp[sum];
	}
}