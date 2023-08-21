#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>

using namespace std;

int main()
{
	// Random training sets for XOR -- two inputs and one output
	freopen("trainingData.txt","w",stdout);
	cout << "topology: 4 4 3" << endl;
	for(int i = 10000; i >= 0; --i)
	{
		vector<int> pict(4,0);
		int n1 = rand() % 4;
		int n2 = rand() % 4;
		while(n1 == n2) n2 = rand() % 4;

		pict[n1] = 1;
		pict[n2] = 1;
		cout << "in: ";
		string str = "";
		for(int i = 0; i < (int)pict.size(); i++)
		{
			str += to_string(pict[i]);
			cout << pict[i] << ".0 ";
		}
		cout << endl;
		cout << "out: ";
		if(str == "1100" || str == "0011") cout << "1.0 0.0 0.0";
		if(str == "1010" || str == "0101") cout << "0.0 1.0 0.0";
		if(str == "1001" || str == "0110") cout << "0.0 0.0 1.0";
		cout << endl;
	}

}
