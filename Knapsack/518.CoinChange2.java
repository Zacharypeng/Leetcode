/*
	dp[i][j] means whether sum j can be formed from the first ith element in nums
	here, wo CAN reuse element
	transformation fuction: dp[i][j] = dp[i-1][j] || dp[i][j-num]

	Notice:
		BASE CASE!
		different index problem between dp and coins!!
*/
class Solution {
	public int change(int amount, int[] coins) {
		int[][] dp = new int[coins.length+1][amount+1];
		dp[0][0] = 1;
		for (int i = 1; i < coins.length; i++) {
			dp[i][0] = 1;
			for (int j = 1; j <= amount; j++) {
				dp[i][j] = dp[i-1][j] + (j >= coins[i-1] ? dp[i][j-coins[i-1]] : 0);
			}
		}
		return dp[coins.length][amount];
	}
}