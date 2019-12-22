/*
	364. Nested List Weight Sum II
	Input: [[1,1],2,[1,1]]
	Output: 8 
	Explanation: Four 1's at depth 1, one 2 at depth 2.

	Note: BFS. Keep adding the previous int when entering each level
*/
public int depthSumInverse(List<NestedInteger> nestedList) {        
    int newInt = 0, sum = 0;        
    while (!nestedList.isEmpty()) {
        List<NestedInteger> nextLevel = new ArrayList<>();
        for (NestedInteger cur: nestedList) {
            if (cur.isInteger()) newInt += cur.getInteger();
            else nextLevel.addAll(cur.getList());
        }
        sum += newInt;
        nestedList = nextLevel;
    }
    return sum;
}	

/*
    244. Shortest Word Distance II

    Note: compare the index in the given list, not the loop idx
*/
class WordDistance {
    Map<String, List<Integer>> map;
    public WordDistance(String[] words) {
        this.map = new HashMap<>();
        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            map.putIfAbsent(word, new ArrayList<Integer>());
            map.get(word).add(i);
        }
    }
    
    public int shortest(String word1, String word2) {
        List<Integer> list1 = map.get(word1);
        List<Integer> list2 = map.get(word2);
        int dis = Integer.MAX_VALUE;
        for (int i = 0, j = 0; i < list1.size() && j < list2.size(); ) {
            int temp = Math.abs(list1.get(i) - list2.get(j));
            dis = Math.min(temp, dis);
            if (list1.get(i) > list2.get(j)) j++;
            else i++;
        }
        return dis;
    }
}    

/*
	366. Find Leaves of Binary Tree
	Input: [1,2,3,4,5]
  
          1
         / \
        2   3
       / \     
      4   5    

	Output: [[4,5,3],[2],[1]]	

	Note:
		Find the right position in List and insert
        construct the depth from the bottom to top
*/
class Solution {
    public List<List<Integer>> findLeaves(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();
        helper(root, ans);
        return ans;
    }
    
    private int helper(TreeNode node, List<List<Integer>> ans) {
        if (node == null) return -1;
        int depth = 1 + Math.max(helper(node.left, ans), helper(node.right, ans));
        if (depth >= ans.size()) ans.add(new ArrayList<Integer>());
        ans.get(depth).add(node.val);
        return depth;
    }
}

/*
	716. Max Stack

	Note:
		Extra stack to store the max number at every time stamp
*/
class MaxStack {
    
    Stack<Integer> stack;
    Stack<Integer> maxStack;

    /** initialize your data structure here. */
    public MaxStack() {
        stack = new Stack<>();
        maxStack = new Stack<>();
    }
    
    public void push(int x) {
        int tempMax = maxStack.isEmpty() ? Integer.MIN_VALUE : maxStack.peek();
        if (tempMax > x) maxStack.push(tempMax);
        else maxStack.push(x);
        stack.push(x);
    }
    
    public int pop() {
        maxStack.pop();
        return stack.pop();
    }
    
    public int top() {
        return stack.peek();
    }
    
    public int peekMax() {
        return maxStack.peek();
    }
    
    public int popMax() {
        int max = maxStack.peek();
        Stack<Integer> tempStack = new Stack<>();
        while (stack.peek() != max) {
            tempStack.push(stack.pop());
            maxStack.pop();
        }
        stack.pop();
        maxStack.pop();
        while (!tempStack.isEmpty()) {
            push(tempStack.pop());
        }
        return max;
    }
}

/*
	987. Vertical Order Traversal of a Binary Tree

	Note:
		BFS. Notice the Comparator of the Priority Queue
*/
class Solution {

    class Point {
        int x, y, val;
        Point (int x, int y, int val) {
            this.x = x;
            this.y = y;
            this.val = val;
        }
    }

    public List<List<Integer>> verticalTraversal(TreeNode root) {
        PriorityQueue<Point> pq = new PriorityQueue<>(new Comparator<Point>() {
            public int compare(Point p1, Point p2) {
                if (p1.x != p2.x) return p1.x - p2.x;
                else if (p1.y != p2.y) return p2.y - p1.y;
                else return p1.val - p2.val;
            }
        });
        dfs(pq, root, 0, 0);
        List<List<Integer>> ans = new ArrayList<>();
        int prev = Integer.MIN_VALUE;
        while (!pq.isEmpty()) {
            Point p = pq.poll();
            if (p.x > prev) {
                List<Integer> temp = new ArrayList<>();
                temp.add(p.val);
                ans.add(temp);
            } else {
                List<Integer> list = ans.get(ans.size() - 1);
                list.add(p.val);
            }
            prev = p.x;
        }
        return ans;
    }

    private void dfs(PriorityQueue<Point> pq, TreeNode node, int x, int y) {
        if (node == null) return;
        pq.add(new Point(x, y, node.val));
        dfs(pq, node.left, x - 1, y - 1);
        dfs(pq, node.right, x + 1, y - 1);
    }
}

/*
	605. Can Place Flowers
	Input: flowerbed = [1,0,0,0,1], n = 1
	Output: True

	Input: flowerbed = [1,0,0,0,1], n = 2
	Output: False
*/
public boolean canPlaceFlowers(int[] f, int n) {
    int cnt = 0;
    for (int i = 0; i < f.length && cnt < n; i++) {
        if (f[i] == 0) {
            int pre = i == 0 ? 0 : f[i-1];
            int next = i == f.length - 1 ? 0 : f[i+1];
            if (pre == 0 && next == 0) {
                f[i] = 1;
                cnt++;
            }
        }            
    }
    return cnt == n;
}

/*
	256. Paint House
	Input: [[17,2,17],[16,16,5],[14,3,19]]
	Output: 10
	Explanation: Paint house 0 into blue, paint house 1 into green, paint house 2 into blue. 
	             Minimum cost: 2 + 5 + 3 = 10.
*/
public int minCost(int[][] costs) {
    int r = 0, b = 0, g = 0;
    for (int i = 0; i < costs.length; i++) {
        int tempR = r;
        r = Math.min(b, g) + costs[i][0];
        int tempG = g;
        g = Math.min(tempR, b) + costs[i][2];
        b = Math.min(tempR, tempG) + costs[i][1];
    }
    return Math.min(r, Math.min(b, g));
}	             

/*
    254. Factor Combinations
    Input: 12
    Output:
    [
      [2, 6],
      [2, 2, 3],
      [3, 4]
    ]

    Note: Remember to remove the last element before go into the dfs
            Also remember to remove the last element after the dfs
*/
class Solution {
    public List<List<Integer>> getFactors(int n) {
        List<List<Integer>> ans = new ArrayList<>();
        helper(2, n, ans, new ArrayList<>());
        return ans;
    }
    
    private void helper(int start, int n, List<List<Integer>> ans, List<Integer> temp) {        
        for (int i = start; i * i <= n; i++) {
            if (n % i == 0) {
                temp.add(i);
                temp.add(n / i);
                ans.add(new ArrayList<>(temp));
                temp.remove(temp.size() - 1);
                helper(i, n / i, ans, temp);
                temp.remove(temp.size() - 1);
            }
        }
    }
}

/*
    373. Find K Pairs with Smallest Sums
    Input: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
    Output: [[1,2],[1,4],[1,6]] 

    Notice:
        Remeber to check whether priority queue is empty! because k could be bigger than the total size!
        Rember to check whether the int array's length is 0
*/
class Solution {
    class Pair {
        int x, y;
        Pair(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        PriorityQueue<Pair> pq = new PriorityQueue<>((a, b) -> (nums1[a.x] + nums2[a.y] - nums1[b.x] - nums2[b.y]));
        List<List<Integer>> ans = new ArrayList<>();
        if (nums1.length == 0 || nums2.length == 0) return ans;
        for (int i = 0; i < nums1.length && i < k; i++)
            pq.add(new Pair(i, 0));
        while (k-- > 0 && !pq.isEmpty()) {            
            Pair p = pq.poll();
            List<Integer> temp = new ArrayList<>(Arrays.asList(nums1[p.x], nums2[p.y]));            
            ans.add(temp);
            if (p.y + 1 < nums2.length)
                pq.add(new Pair(p.x, p.y + 1));
        }
        return ans;
    }
}


/*
    265. Paint House II
    Input: [[1,5,3],[2,9,4]]
    Output: 5

    Note: calculation value at every entry, and update min, secMin and minIdx
*/
public int minCostII(int[][] costs) {
    int preMinIdx = -1, preSecMin = 0, preMin = 0;
    for (int i = 0; i < costs.length; i++) {
        int min = Integer.MAX_VALUE, secMin = Integer.MAX_VALUE;
        int minIdx = -1;
        for (int j = 0; j < costs[0].length; j++) {
            int val = costs[i][j] + (j == preMinIdx ? preSecMin : preMin);
            if (minIdx < 0) {
                min = val;
                minIdx = j;
            } else if (val < min) {
                secMin = min;
                minIdx = j;
                min = val;
            } else if (val < secMin) {
                secMin = val;
            }
        }            
        preMinIdx = minIdx;
        preSecMin = secMin;
        preMin = min;
    }
    return preMin;
}


/*
    730. Count Different Palindromic Subsequences
    Input: 
    S = 'bccb'
    Output: 6 (Sebsequence)

    Note:
        the right boudary has to be strictly less than end
*/
class Solution {
    int div=1000000007;
    public int countPalindromicSubsequences(String S) {
        TreeSet[] chars = new TreeSet[26];
        for (int i = 0; i < 26; i++)
            chars[i] = new TreeSet<Integer>();
        for (int i = 0; i < S.length(); i++) {
            int idx = S.charAt(i) - 'a';
            chars[idx].add(i);
        }
        return memo(chars, new Integer[S.length() + 1][S.length() + 1], S, 0, S.length());
    }

    private int memo(TreeSet<Integer>[] chars, Integer[][] dp, String S, int start, int end) {
        if (start >= end) return 0;
        if (dp[start][end] != null) return dp[start][end];
        long ans = 0;
        for (int i = 0; i < 26; i++) {
            Integer newStart = chars[i].ceiling(start);
            Integer newEnd = chars[i].lower(end);
            if (newStart == null || newStart >= end) continue;
            ans++; // for a single character
            if (newStart != newEnd) ans++; // for a pair
            ans += memo(chars, dp, S, newStart + 1, newEnd);
        }
        dp[start][end] = (int) (ans % div);
        return dp[start][end];
    }
}    


/*
    636. Exclusive Time of Functions

    Note: 
        store id in the stack
        when meets an end, remember to +1 for duration ans pre
*/    
public int[] exclusiveTime(int n, List<String> logs) {
    int[] time = new int[n];
    Stack<Integer> stack = new Stack<>();
    int pre = 0;
    for (String log: logs) {
        String[] strs = log.split(":");
        int cur = Integer.parseInt(strs[2]);
        int id = Integer.parseInt(strs[0]);
        if (strs[1].equals("start")) {
            if (!stack.isEmpty()) {
                time[stack.peek()] += cur - pre;
            }
            stack.push(id);
            pre = cur;
        } else {
            time[stack.pop()] += cur - pre + 1;
            pre = cur + 1;
        }
    }
    return time;
}        

/*
    152. Maximum Product Subarray
*/
public int maxProduct(int[] nums) {
    int min=nums[0], max=nums[0], ans=nums[0];
    for(int i=1; i<nums.length; i++) {
        if(nums[i]<0) {
            int temp = min;
            min = max;
            max = temp;
        }
        min = Math.min(min*nums[i], nums[i]);
        max = Math.max(max*nums[i], nums[i]);
        ans = Math.max(ans, max);
    }
    return ans;
}    

/*
    72. Edit Distance

    Notice:
        initialize the dp: when one string is 0 and the other
        the index difference between string and dp array!!
*/
public int minDistance(String word1, String word2) {
    int m = word1.length();
    int n = word2.length();
    int[][] dp = new int[m + 1][n + 1];
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1.charAt(i - 1) == word2.charAt(j - 1))
                dp[i][j] = dp[i - 1][j - 1];
            else
                dp[i][j] = Math.min(Math.min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]) + 1;
        }
    }
    return dp[m][n];
}

/*
    57. Insert Interval

    Note:
        edge case in the second loop condition: the newInterger's end can be equal to the integer's start
*/        
public int[][] insert(int[][] intervals, int[] newInterval) {
    List<int[]> ans = new ArrayList<>();
    int idx = 0;
    while (idx < intervals.length && newInterval[0] > intervals[idx][1])
        ans.add(intervals[idx++]);
    while (idx < intervals.length && newInterval[1] >= intervals[idx][0]) {
        int start = Math.min(newInterval[0], intervals[idx][0]);
        int end = Math.max(newInterval[1], intervals[idx][1]);
        newInterval = new int[]{start, end};
        idx++;
    }
    ans.add(newInterval);
    while (idx < intervals.length)
        ans.add(intervals[idx++]);
    int[][] res = new int[ans.size()][2];
    for (int i = 0; i < ans.size(); i++) {
        res[i][0] = ans.get(i)[0];
        res[i][1] = ans.get(i)[1];
    }            
    return res;
}    


/*
    698. Partition to K Equal Sum Subsets
    Input: nums = [4, 3, 2, 3, 5, 2, 1], k = 4
    Output: True
    Explanation: It's possible to divide it into 4 subsets (5), (1, 4), (2,3), (2,3) with equal sums.

    Notice:
        need numOfElement!! when target == sum == 0, have to check. nums can have negative numbers
*/
class Solution {
    public boolean canPartitionKSubsets(int[] nums, int k) {
        int sum = 0;
        for (int num: nums)
            sum += num;
        if (sum % k != 0) return false;
        int target = sum / k;
        boolean[] visited = new boolean[nums.length];
        int start = 0, numOfElement = 0;
        return helper(target, k, nums, visited, start, numOfElement, 0);        
    }
    
    private boolean helper(int target, int k, int[] nums, boolean[] visited, int start, int numOfElement, int sum) {
        if (k == 1) return true;
        if (sum == target && numOfElement > 0) return helper(target, k - 1, nums, visited, 0, 0, 0);
        for (int i = start; i < nums.length; i++) {
            if (!visited[i] && sum + nums[i] <= target) {
                visited[i] = true;
                if (helper(target, k, nums, visited, i + 1, numOfElement + 1, sum + nums[i])) return true;
                visited[i] = false;
            }
        }
        return false;
    }
}

/*
    273. Integer to English Words
*/        
class Solution {
    public String[] belowTen = new String[] {"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"};
    public String[] belowTwenty = new String[]{"Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
    public String[] belowHundred = new String[]{"", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
    public String numberToWords(int num) {
        if(num==0) return "Zero";
        return helper(num);
    }
    
    public String helper(int num) {
        String ans = new String();
        if(num<10)
            ans = belowTen[num%10];
        else if(num<20)
            ans = belowTwenty[num%10];
        else if(num<100)
            ans = belowHundred[num/10] + " "+ helper(num%10);
        else if(num<1000)
            ans = helper(num/100) + " Hundred " + helper(num%100);
        else if(num<1000000)
            ans = helper(num/1000) + " Thousand " + helper(num%1000);
        else if(num<1000000000)
            ans = helper(num/1000000) + " Million " + helper(num%1000000);
        else
            ans = helper(num/1000000000) + " Billion " + helper(num%1000000000);
        return ans.trim();
    }
}    


/*
    611. Valid Triangle Number
    Input: [2,2,3,4]
    Output: 3

    Note:
        lock the very last number in nums;
        loop idx from righ to left;
*/
public int triangleNumber(int[] nums) {
    Arrays.sort(nums);
    int ans = 0;
    for (int i = nums.length - 1; i >= 2; i--) {
        int l = 0, r  = i - 1;
        while (l < r) {
            if (nums[r] + nums[l] > nums[i]) {
                ans += r - l;
                r--;
            } else {
                l++;
            }
        }            
    }
    return ans;
}

/*
    149. Max Points on a Line
    Input: [[1,1],[2,2],[3,3]]
    Output: 3

    Note:
        for every points, we must have new map and new overlap
        GCD!
*/
class Solution {
    public int maxPoints(int[][] points) {
        int len = points.length;
        int max = 0;        
        for (int i = 0; i < len; i++) {            
            int overlap = 0, line = 0;
            Map<String, Integer> map = new HashMap<>();
            for (int j = i + 1; j < len; j++) {
                int x = points[i][0] - points[j][0];
                int y = points[i][1] - points[j][1];
                if (x == 0 && y == 0) {
                    overlap++;
                    continue;
                }
                int gcd = getGcd(x, y);
                x = x / gcd;
                y = y / gcd;
                String slope = String.valueOf(x) + String.valueOf(y);
                map.put(slope, map.getOrDefault(slope, 0) + 1);
                line = Math.max(line, map.get(slope));
            }
            max = Math.max(max, line + overlap + 1);
        }
        return max;
    }
    
    private int getGcd(int x, int y) {
        if (y == 0) return x;
        return getGcd(y, x % y);
    }
}        


/*
    380. Insert Delete GetRandom O(1)

    Note:
        always remove the last element of the list, by swapping the last element with the target element
        remember to remove the entry after remove the element
*/
class RandomizedSet {
    
    List<Integer> nums;
    Map<Integer, Integer> map;
    Random rdm;
    
    /** Initialize your data structure here. */
    public RandomizedSet() {
        rdm = new Random();
        nums = new ArrayList<>();
        map = new HashMap<>();
    }
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
        if (map.containsKey(val)) return false;
        map.put(val, nums.size());
        nums.add(val);
        return true;
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
        if (!map.containsKey(val)) return false;
        int idx = map.get(val);
        if (idx != nums.size() - 1) {
            int lastNum = nums.get(nums.size() - 1);
            nums.set(idx, lastNum);   
            map.put(lastNum, idx);
        }
        nums.remove(nums.size() - 1);
        map.remove(val);
        return true;
    }
    
    /** Get a random element from the set. */
    public int getRandom() {
        return nums.get(rdm.nextInt(nums.size()));
    }
}


/*
    50. Pow(x, n)

    Notice:
        difference between float and double and long !!!
        int power index could overflow!! 
        when casting to long, always surround the target value with parenthese!!
*/
public double myPow(double x, int n) {
    if (n == 0) return 1;
    if (n < 0) x = 1 / x;        
    long p = Math.abs((long) n);
    return p % 2 == 1 ? x * myPow(x * x, (int) (p / 2)) : myPow(x * x, (int) (p / 2));
}        


/*
    277. Find the Celebrity

    Notice:
        it's true that we only need to check [0, candidate)
        but we still have to go into the second looop for checking
        if candidate == 0, then there is not chance to enter the loop, if we loop in [0, candidate)
*/
public int findCelebrity(int n) {
    int candidate = 0;
    for (int i = 1; i < n; i++) {
        if (knows(candidate, i))
            candidate = i;
    }
    for (int i = 0; i < n; i++) {
        if (i != candidate && (knows(candidate, i) || !knows(i, candidate)))
            return -1;
        if (i > candidate) return candidate;
    }
    return candidate;
}        


/*
    297. Serialize and Deserialize Binary Tree

    Notice:
        for serialize, traditional bfs; continue when hitting null, otherwise offer node into the queue
        string comparasion use equals!!!
*/
public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if (root == null) return "";
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        StringBuilder sb = new StringBuilder();
        while (!q.isEmpty()) {
            TreeNode node = q.poll();
            if (node == null) {
                sb.append("n ");
                continue;
            }         
            sb.append(node.val + " ");
            q.offer(node.left);
            q.offer(node.right);
        }        
        return sb.toString().trim();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data.length() == 0) return null;
        String[] strs = data.split(" ");
        Queue<TreeNode> q = new LinkedList<>();
        TreeNode root = new TreeNode(Integer.parseInt(strs[0]));
        q.offer(root);
        for (int i = 1; i < strs.length; i++) {
            TreeNode node = q.poll();
            if (!strs[i].equals("n")) {
                int leftVal = Integer.parseInt(strs[i]);
                TreeNode left = new TreeNode(leftVal);
                node.left = left;
                q.offer(left);
            }            
            i++;
            if (!strs[i].equals("n")) {
                int rightVal = Integer.parseInt(strs[i]);
                TreeNode right = new TreeNode(rightVal);
                node.right = right;
                q.offer(right);
            }
        }
        return root;
    }
}

/*
    56. Merge Intervals
    Input: [[1,3],[2,6],[8,10],[15,18]]
    Output: [[1,6],[8,10],[15,18]]
*/
public int[][] merge(int[][] intervals) {
    if (intervals.length <= 1)
        return intervals;

    // Sort by ascending starting point
    Arrays.sort(intervals, (i1, i2) -> Integer.compare(i1[0], i2[0]));

    List<int[]> result = new ArrayList<>();
    int[] newInterval = intervals[0];
    result.add(newInterval);
    for (int[] interval : intervals) {
        if (interval[0] <= newInterval[1]) // Overlapping intervals, move the end if needed
            newInterval[1] = Math.max(newInterval[1], interval[1]);
        else {                             // Disjoint intervals, add the new interval to the list
            newInterval = interval;
            result.add(newInterval);
        }
    }

    return result.toArray(new int[result.size()][]);
}


/*
    341. Flatten Nested List Iterator
    Input: [[1,1],2,[1,1]]
    Output: [1,1,2,1,1]

    Notice:
        increment cnt only AFTER getting next number!!
*/
public class NestedIterator implements Iterator<Integer> {

    List<Integer> list;
    int cnt;
    public NestedIterator(List<NestedInteger> nestedList) {
        list = new ArrayList<>();
        cnt = 0;
        helper(list, nestedList);
    }
    
    private void helper(List<Integer> list, List<NestedInteger> nl) {
        for (NestedInteger n: nl) {
            if (n.isInteger()) list.add(n.getInteger());
            else helper(list, n.getList());
        }
    }

    @Override
    public Integer next() {        
        Integer n =  cnt < list.size() ? list.get(cnt) : null;
        cnt++;
        return n;
    }

    @Override
    public boolean hasNext() {
        return cnt < list.size();
    }
}


/*
    261. Graph Valid Tree
    Input: n = 5, and edges = [[0,1], [0,2], [0,3], [1,4]]
    Output: true

    Note:
        keep union vertices. So for every new encountered vertices, them shouldn't belong to the same set.
        have to check whether len(edges) == len(vertices) !!
*/        
class Solution {
    public boolean validTree(int n, int[][] edges) {
        int[] parents = new int[n];
        for (int i = 0; i < n; i++)
            parents[i] = i;
        for (int[] e: edges) {
            int v1 = find(parents, e[0]);
            int v2 = find(parents, e[1]);
            if (v1 == v2) return false;
            parents[v1] = v2;
        }
        return edges.length == n - 1;
    }
    
    private int find(int[] p, int v) {
        if (p[v] == v) return v;
        return find(p, p[v]);
    }
}        


/*
    1028. Recover a Tree From Preorder Traversal
    Input: "1-2--3--4-5--6--7"
    Output: [1,2,5,3,4,6,7]

    Note:
        number of - means the level. 
        if the stack size is bigger than the depth, it means the node at the peek is already constructed.

    Notice:
        when calculating the val, it's not += !!
*/
public TreeNode recoverFromPreorder(String S) {
    Stack<TreeNode> stack = new Stack<>();
    for (int i = 0; i < S.length(); ) {
        int level, val;
        for (level = 0; i < S.length() && S.charAt(i) == '-'; i++)
            level++;
        for (val = 0; i < S.length() && S.charAt(i) != '-'; i++)
            val = val * 10 + Integer.parseInt(S.charAt(i) + "");
        while (stack.size() > level)
            stack.pop();
        TreeNode node = new TreeNode(val);
        if (stack.size() > 0) {
            if (stack.peek().left == null)
                stack.peek().left = node;
            else stack.peek().right = node;
        }                
        stack.push(node);
    }
    while (stack.size() > 1)
        stack.pop();
    return stack.pop();
}        



/*
    655. Print Binary Tree
    Input:
     1
    /
   2
    Output:
    [["", "1", ""],
     ["2", "", ""]]
*/
class Solution {
    public List<List<String>> printTree(TreeNode root) {
        List<List<String>> ans = new ArrayList<>();
        int height = find_height(root, 0);      
        int length = (int)Math.pow(2, height)-1;
        for(int line = 0; line < height; line++) {
            List<String> row = new ArrayList<>();
            for(int i = 0; i < length; i++)
                row.add("");
            ans.add(row);
        }
        fill_tree(root, ans, 0, length-1, height, 0);
        return ans;
    }
    
    private int find_height(TreeNode node, int level) {
        if(node == null) return level;
        return Math.max(find_height(node.left, level + 1), find_height(node.right, level + 1));
    }
    
    private void fill_tree(TreeNode node, List<List<String>> ans, int left, int right, int height, int level) {
        if(height == level || node == null) return;
        int mid = left - (left - right) / 2;
        ans.get(level).set(mid, String.valueOf(node.val));
        fill_tree(node.left, ans, left, mid - 1, height, level + 1);
        fill_tree(node.right, ans, mid + 1, right, height, level + 1);
    }
}     


/*
    516. Longest Palindromic Subsequence

    Note:
        outer loop from right to left; inner loop from left to right; (vice versa)
        transition table
*/
public int longestPalindromeSubseq(String s) {
    int len = s.length();
    int[][] dp = new int[len][len];
    for (int i = len - 1; i >= 0; i--) {
        dp[i][i] = 1;
        for (int j = i + 1; j < len; j++) {
            if (s.charAt(i) == s.charAt(j))
                dp[i][j] = dp[i+1][j-1] + 2;
            else dp[i][j] = Math.max(dp[i+1][j], dp[i][j-1]);
        }
    }
    return dp[0][len-1];
}            


/*
    127. Word Ladder
    Input:
    beginWord = "hit",
    endWord = "cog",
    wordList = ["hot","dot","dog","lot","log","cog"]

    Output: 5

    Note:
        check whether wordList contains endWord at the beginning!!
        two-end BFS
*/
public int ladderLength(String beginWord, String endWord, List<String> wordList) {
    Set<String> end = new HashSet<>();
    Set<String> begin = new HashSet<>();
    Set<String> visited = new HashSet<>();
    begin.add(beginWord);
    end.add(endWord);
    Set<String> dict = new HashSet<>();
    dict.addAll(wordList);
    if (!dict.contains(endWord)) return 0;
    int len = 1;
    while (!begin.isEmpty() && !end.isEmpty()) {
        if (begin.size() > end.size()) {
            Set<String> t = begin;
            begin = end;
            end = t;
        }
        Set<String> temp = new HashSet<>();
        int size = begin.size();
        for (String s: begin) {
            char[] ch = s.toCharArray();
            for (int i = 0; i < ch.length; i++) {
                char oldChar = ch[i];
                for (char c = 'a'; c <= 'z'; c++) {
                    ch[i] = c;
                    String target = String.valueOf(ch);
                    if (end.contains(target)) return len + 1;
                    if (!visited.contains(target) && dict.contains(target)) {
                        temp.add(target);
                        visited.add(target);
                    }
                    ch[i] = oldChar;
                }
            }
        }
        len++;
        begin = temp;
    }
    return 0;
}        


/*
    126. Word Ladder II
    Input:
    beginWord = "hit",
    endWord = "cog",
    wordList = ["hot","dot","dog","lot","log","cog"]

    Output:
    [
      ["hit","hot","dot","dog","cog"],
      ["hit","hot","lot","log","cog"]
    ]

    Note:
        when to remove candidate list
        the distance map is to record shortest distance and can be served as a visited set
        for getting neighbors, check if the changed char is equal to the original char
*/
class Solution {
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        Map<String, Set<String>> neighbors = new HashMap<>();
        Queue<String> q = new LinkedList<>();
        Map<String, Integer> distance = new HashMap<>();
        q.offer(beginWord);
        distance.put(beginWord, 0);
        Set<String> set = new HashSet<>();
        set.addAll(wordList);
        List<List<String>> ans = new ArrayList<>();
        boolean found = false;
        int step = 0;
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                String cur = q.poll();
                Set<String> curNeighbors = getNeighbors(cur, set);
                neighbors.put(cur, curNeighbors);
                int curStep = distance.get(cur);
                for (String n: curNeighbors) {                    
                    if (!distance.containsKey(n)) {
                        distance.put(n, curStep + 1);
                        if (n.equals(endWord))
                            found = true;
                        else q.add(n);
                    }
                }
            }
            if (found) break;
        }
        
        dfs(beginWord, endWord, set, distance, neighbors, ans, new ArrayList<>());
        return ans;
    }
    
    private void dfs(String start, String end, Set<String> set, Map<String, Integer> distance, Map<String, Set<String>> neighbors, List<List<String>> ans, List<String> curList) {
        if (start.equals(end)) {
            curList.add(end);
            ans.add(new ArrayList<>(curList));
            curList.remove(curList.size() - 1);
            return;
        }
        curList.add(start);
        if (neighbors.containsKey(start)) {
            for (String n: neighbors.get(start)) {
                if (distance.get(n) == distance.get(start) + 1)
                    dfs(n, end, set, distance, neighbors, ans, curList);            
            }            
        }    
        curList.remove(curList.size() - 1); 
    }
    
    private Set<String> getNeighbors(String str, Set<String> set) {
        char[] ch = str.toCharArray();
        Set<String> neighbors = new HashSet<>();
        for (int i = 0; i < str.length(); i++) {
            char oldChar = ch[i];
            for (char c = 'a'; c <= 'z'; c++) {
                ch[i] = c;
                if (c == oldChar) continue;
                String target = String.valueOf(ch);
                if (!target.equals(str) && set.contains(target))
                    neighbors.add(target);
                ch[i] = oldChar;
            }
        }
        return neighbors;
    }
}    



/*
    133. Clone Graph

    Notice:
        put the newNode, right after it has been initiated!
*/
/*
// Definition for a Node.
class Node {
    public int val;
    public List<Node> neighbors;

    public Node() {}

    public Node(int _val,List<Node> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};
*/
class Solution {
    
    Map<Integer, Node> map = new HashMap<>();
    
    public Node cloneGraph(Node node) {
        if (node == null) return node;
        return clone(node);        
    }
    
    private Node clone(Node node) {
        if (node == null) return null;
        if (map.containsKey(node.val)) return map.get(node.val);
        Node n = new Node(node.val, new ArrayList<Node>());
        map.put(n.val, n);
        for (Node next: node.neighbors) {
            n.neighbors.add(clone(next));
        }        
        return n;
    }
}        





