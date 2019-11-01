/*
    Think of dp as a 2D matrix, try to start from [0,0] to [m,n]
    dp[i][j] is true when the dp[i-1][j] or dp[i][j-1] is true
    Notice 
        the base case where dp[0][0] = true
        the different index between String and dp
*/
class Solution {
    public boolean isInterleave(String s1, String s2, String s3) {        
        int m = s1.length(), n = s2.length();
        if (m + n != s3.length()) return false;
        boolean[][] dp = new boolean[m+1][n+1];
        dp[0][0] = true;
        for (int i = 1; i <= m; i++)
            dp[i][0] = dp[i-1][0] && s1.charAt(i-1) == s3.charAt(i-1);
        for (int i = 1; i <= n; i++)
            dp[0][i] = dp[0][i-1] && s2.charAt(i-1) == s3.charAt(i-1);
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = (dp[i-1][j] && s1.charAt(i-1) == s3.charAt(i+j-1)) ||
                            (dp[i][j-1] && s2.charAt(j-1) == s3.charAt(i+j-1));
            }
        }
        return dp[m][n];
    }
}