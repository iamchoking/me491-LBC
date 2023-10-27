#include <iostream>
#include "20190673.hpp"

//========== SKELETON CODE ==========
int main() {
    Eigen::Vector<int, 12> state;
    state << 0,0,0,0,0,0,0,0,0,0,0,0; // 3.97701
    // state << 1,0,0,0,0,0,0,0,0,0,0,0; // 3.51715
    //state << 1,1,0,0,0,0,0,0,0,0,0,0; // 3.98413

    // state << 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0; // 3.2 & 10 (??)
    // state << 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0; // 3.33333 & 10
    // state << 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1; // 2.33333 & 4
    // state << 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0; // 4 & 7
    // state << 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0; // 3.33333 & 1

    //Sanity Checks
    // state << 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0; // must be 4
    //state << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1; // must be 0


    //auto sString = dabString(state);
    //auto actions = actionsFrom(sString);
    //cout << dabShow(sString,'\n')<<endl;
    //cout << actions << endl;
    //cout << size(actions) << " actions possible" << endl;
    //
    //takeAction(sString,actions[3]);
    //cout << dabShow(sString,'\n')<<endl;


    //cout << state << endl;

    std::cout << "optimal value for the state: " << getOptimalValue(state) << std::endl;
    std::cout << "optimal action for the state: " << getOptimalAction(state) << std::endl;
    return 0;
}
