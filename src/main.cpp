#include <iostream>
//#include "20190673.hpp"
//==================20190673.hpp=================
#include <map>
#include <chrono>
#include <utility>
#include <string>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <random>

using namespace Eigen;
using namespace std;

/// discount factor
double GAMMA = 1.00;

ostream& operator << (ostream& os, vector<int>& P);

// pure functions

/// parses an Eigen::Vector<int, 12> object into a std::string of length 9
string dabString(const Eigen::Vector<int, 12>& data){
    string dab = "____________";
    int idx = 0;
    for(auto row:data.rowwise()){
        for(auto e:row){
            if(e == 1){dab[idx] = '-';}
            idx += 1;
        }
    }
    return dab;
}


/// internal helper function to make dab-strings more readable
//  -- --    0  1
// |  |  |  6  8  10
//  -- --    2   3
// |  |  |  7  9  11
//  -- --    4   5

//  -- -- $|  |  |$ -- -- $|  |  |$ -- --

string dabShow(const string& dabString, const char& delimiter){
    string l1 = " -- -- ";
    string l2 = "|  |  |";
    string l3 = " -- -- ";
    string l4 = "|  |  |";
    string l5 = " -- -- ";

    if(delimiter == '\n'){
        l1 += "  0  1   ";
        l2 += " 6  8  10";
        l3 += "  2   3  ";
        l4 += " 7  9  11";
        l5 += "  4   5  ";
    }

    if(dabString[ 0]  == '_'){l1[1] = ' ';l1[2] = ' ';}
    if(dabString[ 1]  == '_'){l1[4] = ' ';l1[5] = ' ';}
    if(dabString[ 2]  == '_'){l3[1] = ' ';l3[2] = ' ';}
    if(dabString[ 3]  == '_'){l3[4] = ' ';l3[5] = ' ';}
    if(dabString[ 4]  == '_'){l5[1] = ' ';l5[2] = ' ';}
    if(dabString[ 5]  == '_'){l5[4] = ' ';l5[5] = ' ';}

    if(dabString[ 6]  == '_'){l2[0] = ' ';}
    if(dabString[ 7]  == '_'){l4[0] = ' ';}
    if(dabString[ 8]  == '_'){l2[3] = ' ';}
    if(dabString[ 9]  == '_'){l4[3] = ' ';}
    if(dabString[10]  == '_'){l2[6] = ' ';}
    if(dabString[11]  == '_'){l4[6] = ' ';}

    return(l1+delimiter+l2+delimiter+l3+delimiter+l4+delimiter+l5);
}

/// extracting possible actions from a given string (return: a vector of indices of '_')
vector<int> actionsFrom(const string& dab){
    vector<int> actions;
    int idx;
    for(idx = 0; idx < dab.length();idx++)
        if(dab[idx] == '_'){actions.push_back(idx);}
    return actions;
}

/// checks the status of board given the string (true: terminal, false: non-terminal)
bool check(const string& dab){
    return dab == "------------";
}

class State{
public:
    /// the string of length 12 that holds the board info.
    string dataString;
    // the number of "boxes" that are already closed
    int numClosed;
    /// pointer to hash table of State objects
    map<string, State*> *stateMap;

    /// vector of possible actions (action "a" means putting "O" on index a)
    vector<int> actions;

    /// vector of possible state transitions for each action (vector in index i holds possible transitions for actions[i])
    vector<vector<State*>>  transS;
    /// vector of rewards for each action (vector in index i holds possible rewards for actions[i])
    vector<vector<double>>  transR;
    /// vector of state transitions probabilities for each action (transP[i][j] correspond to the prob. that actions[i] will transition to transS[i][j])
    vector<vector<double>>  transP;

    /// state value function (populated in policy evaluation loop)
    double v;

    /// state-action value function (q) (for actions[i]) (populated in policy improvement loop)
    vector<double> q;
    /// policy at current state (an index of actions) (populated in policy improvement loop)
    int policy;

    /// terminal state identifier (
    bool terminal;

    /// status of board: O/X/D: Agent/Opponent/Nobody wins (terminal) -:non-terminal
    //    char status;

    /// metadata for state
    bool parsed;
    /// metadata for state
    bool connected;
    /// metadata for state
    bool converged;

    /// log file stream
    ostream& outStream;

    /// constructor (when data is provided as a 12-char string)
    State(string data, map<string, State*> *m,ostream& os): outStream(os){
        dataString = std::move(data);
        stateMap = m;

        terminal = check(dataString);
        numClosed = 0;
        countClosed(); // later used to determine rewards


        if(!terminal){

            parsed = false;
            connected = false;
            converged = false;
            policy = 0; // randomized later
            v = 0;
        }
        else{
            parsed = true;
            connected = true;
            converged = true;
            policy = -1; // policy doesn't matter.
            v = 0;
        }
    }

    /// constructor (when data is provided as Eigen::Vector<int, 12>) (delegated)
    State(const Eigen::Vector<int, 12>& data, map<string, State*> *m,ostream& os): State(dabString(data),m,os){}

    bool operator== (const State &s) const{return (dataString == s.dataString);}
    bool operator== (const string &m) const{return (dataString == m);}
    friend ostream& operator << (ostream& os, const State& s);
    friend ostream& operator << (ostream& os, vector<int>& P);

    /// more detailed output for full data analysis
    void verbose(ostream& os){
        os << "["<<this<<"]:\n" << dabShow(dataString,'\n') <<endl;
        if(terminal){
            os<< "[TERMINAL STATE]"<<endl;
            return;
        }
        os << "Actions: " << actions;
        os << endl;
        os << "Transitions: " << endl;
        for(int idx = 0;idx < actions.size();idx++){
            os << "\tAction[" << idx << "]: " << actions[idx]<<endl;
            os << "\tPossible Transitions:"<<endl;
            for(int idxx = 0;idxx<transS[idx].size();idxx++){
                os << "\t\t" << *transS[idx][idxx] << " (p: "<<transP[idx][idxx]<<" r: " << transR[idx][idxx] << ")" << endl;
            }
        }
        if(converged){os << "Current policy: " << actions[policy] << " (value: "<< v <<")"<< endl;}
    }

    /// counts how many "boxes" are already closed in this state.
    void countClosed(){
        numClosed = 0;
        if(dataString[0] == '-' && dataString[2] == '-' && dataString[6] == '-' && dataString[8 ] == '-'){numClosed += 1;}
        if(dataString[1] == '-' && dataString[3] == '-' && dataString[8] == '-' && dataString[10] == '-'){numClosed += 1;}
        if(dataString[2] == '-' && dataString[4] == '-' && dataString[7] == '-' && dataString[9 ] == '-'){numClosed += 1;}
        if(dataString[3] == '-' && dataString[5] == '-' && dataString[9] == '-' && dataString[11] == '-'){numClosed += 1;}
    }

    /// reads the dataString to populate [actions] with possible actions, and accordingly populate transX vectors with empty vectors
    void parseActions(){
        actions = actionsFrom(dataString);
        for(int idx = 0;idx < actions.size();idx ++){
            vector<State*> ts;
            vector<double> tr;
            vector<double> tp;
            transS.push_back(ts);
            transR.push_back(tr);
            transP.push_back(tp);
        }
    }

    /// parses possible transformations from a single action (action[i]) -> populates transX[i]
    void parseTrans(int idx){
        int action = actions[idx];

        vector<int> oActions;
        vector<string> nDataStrings;

        string tDataString = dataString;
        tDataString[action] = '-';

        outStream << "[PARSE-TRANS "<<dataString <<"&"<<action<<"] (-> "<< tDataString << ")" << endl;

        // The case when Agent wins or Draws immediately
        if(check(tDataString)){ //is terminal
            outStream << dataString << " > a:" << action <<" (T) -> " << tDataString <<" | ";
            transS[idx].push_back(getStatePtr(tDataString));
            transP[idx].push_back(1);
            transR[idx].push_back( double( (transS[idx].back() -> numClosed) - numClosed) ); // reward is the difference in closed boxes
            return;
        }

        // The case when opponent gets to make a move
        oActions = actionsFrom(tDataString);
        outStream << "Possible Opponent Actions: "<< oActions << endl;

        string nDataString;
        for(auto a:oActions){
            nDataString = tDataString;
            nDataString[a] = '-';
            outStream << dataString << " > a:" << action << "/of:" << (a<10?" ":"") << a << " -> " << nDataString << " | ";
            transS[idx].push_back(getStatePtr(nDataString));
            transP[idx].push_back(1/double(oActions.size()));
            transR[idx].push_back( double( (transS[idx].back() -> numClosed) - numClosed) ); // reward is the difference in closed boxes
        }

    }

    /// parse the data within the state (populate all data)
    void parse(){
        if(parsed){
            //outStream << "[PARSE-SKIP] Skipped" << endl;
            return;
        }
        outStream << "[PARSE] Parsing for ["<<dataString<<"]" << endl;

        parseActions();
        for(int idx = 0;idx < actions.size();idx++){
            parseTrans(idx);
        }

        policy = int(rand()) % size(actions); //initialize with a random policy
        outStream << "[PARSE-PI] Policy Initilized To [" << actions[policy] << "](idx: "<<policy <<")"<<endl;

        parsed = true;

    }

    /// recursively create and parse all data what is "downstream" from current state
    bool connect(){
        parse();
        if(connected){return false;}
        for(const auto& tr:transS){
            for(auto s:tr){
                if(s -> connect()){}
            }
        }
        outStream << "[CONNECT]" << "Connection of [" << dataString << "] Complete." << endl;
        connected = true;
        return true;
    }

    /// reference the hash table to return the State that represents the given data string, or create one if needed.
    [[nodiscard]] State* getStatePtr(const string& newDataString) const{
        State* sp;
        auto tmp = stateMap ->find(newDataString);

        if(stateMap -> find(newDataString) == stateMap -> end()){
            sp = new State(newDataString,stateMap,outStream);
            stateMap -> insert({newDataString,sp});
            //outStream << "[STATE " << dataString << "] Created: " << *sp << endl;
            outStream << "[STATE " << dataString << "] Created: [" << sp << "]" << newDataString << endl;
        }
        else{
            sp = stateMap -> find(newDataString) -> second;
            //outStream << "[STATE " << dataString << "] Linked : " << *sp << endl;
            outStream << "[STATE " << dataString << "] Linked : [" << sp << "]" << newDataString << endl;
        }
        return sp;
    }

    /// calculates the value function (only if "downstream" states are calculated)
    bool evalState(){ // used in the Policy Evaluation Loop

        /// return true if value is already calculated
        if(converged){return true;}

        /* the Value Iteration Method (for reference)
        for(const auto& tr:transS){
            for(auto s:tr){
                if(!(s -> converged)){
                    outStream << "[VI]" << dataString << "not ready! (" << *s << " is not (yet) converged)" << endl;
                    return false;
                }
                // value is not "ready" to be calculated.
            }
        }

        // outStream << this << endl;
        // outStream << dataString << endl;
        // verbose(outStream);

        /// calculate state-action q for each actions[idx]
        for(int idx = 0;idx < actions.size();idx++){
            double temp = 0;
            /// the VI formula is implemented here
            for(int idxx = 0; idxx < transR[idx].size();idxx++){
                temp += transP[idx][idxx]*(transR[idx][idxx] + GAMMA*(transS[idx][idxx] -> v));
            }
            q.push_back(temp);
        }

        /// choose max state-action value as value
        v = *max_element(q.begin(),q.end());

        /// choose corresponding optimal action
        optimalAction = actions[int(max_element(q.begin(),q.end()) - q.begin())];
        // outStream << "optimal action of " << dataString << ":" << optimalAction << endl;
        */

        // Policy iteration method only has to compute the value from the action chosen by the action
        // *policy: an index of action
        // current strategy: make a move on the index actions[policy].

        for(auto s:transS[policy]){
            if(!(s -> converged)){
                outStream << "[VI]" << dataString << "not ready! (" << *s << " is not (yet) converged)" << endl;
                return false;
            }
            // value is not "ready" to be calculated.
        }

        q.clear(); // clear q-values (outdated)

        double temp = 0;
        /// the VI formula is implemented here
        for(int idxx = 0; idxx < transR[policy].size();idxx++){
            temp += transP[policy][idxx]*(transR[policy][idxx] + GAMMA*(transS[policy][idxx] -> v));
        }
        v = temp;

        converged = true;
        return true;
    }

    bool improveState(){
        if(!converged){ // all improvements need to operate on converged v
            return false;
        }

        // calculating q
        double temp;
        for(int idx = 0;idx < size(actions);idx++ ){
            temp = 0;
            /// the VI formula is implemented here
            for(int idxx = 0; idxx < transR[idx].size();idxx++){
                temp += transP[idx][idxx]*(transR[idx][idxx] + GAMMA*(transS[idx][idxx] -> v));
            }
            q.push_back(temp);
        }

        // update policy to argmax
        policy = int(distance(q.begin(), max_element(q.begin(),q.end())));

        // initialize setup for subsequent evaluation.
        converged = false;
        return true;
    }
};



/// overloaded << operator for concise output of State class varables.
ostream & operator<< (ostream& os, const State& s) {
    os << "<" << &s << "> ";
    os << dabShow(s.dataString,'$');
    os << " <" << s.numClosed << "/4>";
    os << " {" << (s.terminal ? 'T':'-') << "}";
    os << " (V:" << s.v << ")";

    if(s.connected){os << "C";}
    if(s.converged){os << "V";}
    return os;
}

/// overloaded << operator for easily viewing vector contents
ostream& operator << (ostream& os, vector<int>& P) {
    for (int i : P)
        os << i << "," ;
    return os;
}

/// overloaded << operator for easily viewing vector contents
ostream& operator << (ostream& os, map<string, State*> m) {
     for (auto const &pair: m) {
         os << "\t{" << pair.first << ": " << *pair.second << "}\n";
     }
    return os;
}

/// Loop to create / populate the states needed for VI
State* stateLoop(const Eigen::Vector<int, 12>& startStateVector,map<string, State*> *stateMap,ostream& os){

    auto startState = new State(startStateVector,stateMap,os);

    stateMap -> insert({startState->dataString,startState});

    os << "[SL] Starting State: " << *startState << endl;

    startState -> connect();

    // startState -> verbose(os);

    os << "[SL-FIN] All relevant states created (size: "<<stateMap -> size()<<"): "<<endl;
    // os << "[SL-FIN] States created. Showing stateMap (size:" << stateMap -> size() << "):" <<endl;

    //os << *(startState -> stateMap) << endl;
    return startState;
}

/// VI loop: iterates over all states in stateMap until all of them converge (usually, it takes only one iteration!)
void valueIterLoop(State* s, const map<string, State*> *stateMap, ostream& os){
    int numStates = int(stateMap -> size());
    os << "\n[VI-START] Value Iteration for " << numStates << " States :" << endl;

    int numConverged = 0;
    int iter = 0;

    while(numConverged < numStates){
        numConverged = 0;
        iter += 1;

        // s -> calcValue();
        for(auto e: *stateMap){
            // os << "checking: [" << e.first <<":"<< stateMap.find(e.first) -> second << "]" << endl;
            // if(e.second == 0){continue;}
            if(e.second -> evalState()){
                // os << *e.second << "has converged" << endl;
                numConverged += 1;
            }
        }
        os << "[VI-ITER" << iter << "] " << numConverged << " out of " << numStates << " converged" << endl;
    }

    os << "[VI-FIN] final statemap (size "<< stateMap -> size() << " ): "<<endl;
    for (auto const &pair: *stateMap) {
        os << "\t{" << pair.first << ": " << *pair.second << "}\n";
    }
}

// DO NOT CHANGE THE NAME AND FORMAT OF THIS FUNCTION
/// final outer loop for VI workflow (executes stateLoop -> valueIterLoop & records elapsed time)
double getOptimalValue(const Eigen::Vector<int, 12>& state){

    auto start = chrono::high_resolution_clock::now();
    auto now = start;
    ofstream ofs("./log.txt");
    ostream& os = ofs;
    //ostream& os = cout;

    map<string, State*> stateMap;

    auto s = stateLoop(state,&stateMap,os);


    auto slDuration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - now);
    now = chrono::high_resolution_clock::now();
    os << "[STATE-LOOP] Finished (elapsed time: " << slDuration.count() <<"us)" << endl;

    s -> verbose(os);
    return 42.0;

    valueIterLoop(s,&stateMap,os);

    auto viDuration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - now);
    os << "[VALUE-ITER] Finished (elapsed time: " << viDuration.count() <<"us)" << endl;

    os << "Final Result (state analysis):"<< endl;
    s ->verbose(os);

    auto totDuration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
    os << "Total Elapsed Time: " << totDuration.count() << "us" << endl;

    return s->v; // return optimal value
}


// SKELETON: HW2
/// DO NOT CHANGE THE NAME AND FORMAT OF THIS FUNCTION
//double getOptimalValue(const Eigen::Vector<int, 12>& state){
//    // return the optimal value given the state
//
//
//    return 42.0;  // return optimal value
//}

/// DO NOT CHANGE THE NAME AND FORMAT OF THIS FUNCTION
int getOptimalAction(const Eigen::Vector<int, 12>& state){
    // return one of the optimal actions given the state.
    // the action should be represented as a state index, at which a line will be drawn.
    /// TODO

    return 0;  // return optimal action
}

//==================20190673.hpp=================

//========== SKELETON CODE ==========
int main() {
    Eigen::Vector<int, 12> state;
    state << 0,0,0,0,0,0,0,0,0,0,0,0;
    //cout << state << endl;

    std::cout << "optimal value for the state: " << getOptimalValue(state) << std::endl;
    //std::cout << "optimal action for the state: " << getOptimalAction(state) << std::endl;
    return 0;
}
