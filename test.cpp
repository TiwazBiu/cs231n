#include <iostream>
#include <unordered_map>
#include <vector>
using namespace std;
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int,int> map;
        for(int i=0;i!=nums.size();++i)
        {
            int complement = target - nums[i];
            if(map.find(complement) != map.end())
                return {i,map.at(complement)};
            map[nums[i]] = i;
        }
        return {};
    }
};

int main()
{
    return 0;
}