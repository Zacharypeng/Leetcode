/*
    dp[i][j] means the shortest substring length of S[:i-1], that its subsequence contains T[:j-1]
    the substring has to be end with S[i-1]

    Notice:
    Base case: every base case has to be big enough in order to record the smaller dp answer
        So for every substring of T, give dp[0][i] = S + 1
*/
class Solution {
    public String minWindow(String S, String T) {
        int s = S.length(), t = T.length();
        int[][] dp = new int[s+1][t+1];
        for (int i = 1; i <= t; i++) dp[0][i] = s + 1;
        for (int i = 1; i <= s; i++) {
            for (int j = 1; j <= t; j++) {
                if (S.charAt(i-1) == T.charAt(j-1)) {
                    dp[i][j] = Math.min(dp[i-1][j], dp[i-1][j-1]) + 1;
                } else {
                    dp[i][j] = dp[i-1][j] + 1;
                }
            }
        }
        int len = Integer.MAX_VALUE, start = -1;
        for (int i = t; i <=s; i++) {
            if (len > dp[i][t]) {
                len = dp[i][t];
                start = i;
            }
        }
        return len >= s + 1 ? "" : S.substring(start - len, start);
    }
}