/////////////////////////////<20190673.hpp>////////////////////////////////
#include <map>
#include <chrono>
#include <utility>
#include <string>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>


using namespace Eigen;
using namespace std;

/// discount factor for value iteration
double GAMMA = 0.98;

ostream& operator << (ostream& os, vector<int>& P);

// pure functions

/// parses an Eigen::Matrix3d object into a std::string of length 9
string tttString(const Matrix3d& data){
    string ttt = "_________";
    int idx = 0;
    for(auto row:data.rowwise()){
        for(auto e:row){
            if(e == 1){ttt[idx] = 'O';}
            else if(e == -1){ttt[idx]= 'X';}
            idx += 1;
        }
    }
    return ttt;
}

/// internal helper function to make ttt-strings more readable
string tttShow(const string& tttString, const char& delimiter){
    return(tttString.substr(0,3) + delimiter+tttString.substr(3,3)+delimiter+tttString.substr(6,3));
}

/// extracting possible actions from a given string (return: a vector of indices of '_')
vector<int> actionsFrom(const string& ttt){
    vector<int> actions;
    int idx;
    for(idx = 0; idx < ttt.length();idx++)
        if(ttt[idx] == '_'){actions.push_back(idx);}
    return actions;
}

/// checks the status of board given the string ('O'/'X': winner decided, 'D': Draw, '-': Game on (non-terminal))
char check(const string& ttt){
    if(ttt[0] == ttt[1] && ttt[1] == ttt[2] && ttt[0] != '_'){return ttt[0];}
    if(ttt[3] == ttt[4] && ttt[4] == ttt[5] && ttt[3] != '_'){return ttt[3];}
    if(ttt[6] == ttt[7] && ttt[7] == ttt[8] && ttt[6] != '_'){return ttt[6];}

    if(ttt[0] == ttt[3] && ttt[3] == ttt[6] && ttt[0] != '_'){return ttt[0];}
    if(ttt[1] == ttt[4] && ttt[4] == ttt[7] && ttt[1] != '_'){return ttt[1];}
    if(ttt[2] == ttt[5] && ttt[5] == ttt[8] && ttt[2] != '_'){return ttt[2];}

    if(ttt[0] == ttt[4] && ttt[4] == ttt[8] && ttt[0] != '_'){return ttt[4];}
    if(ttt[2] == ttt[4] && ttt[4] == ttt[6] && ttt[2] != '_'){return ttt[4];}

    if(actionsFrom(ttt).empty()){return 'D';}
    return '-';
}


class State{
public:
    /// the string of length 9 that holds the board info.
    string dataString;
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

    /// state value
    double value;

    /// state-action (q) value (for actions[i])
    vector<double> values;
    /// optimal action at current state
    int optimalAction;

    /// terminal state identifier
    bool terminal;
    /// status of board: O/X/D: Agent/Opponent/Nobody wins (terminal) -:non-terminal
    char status;

    /// metadata for state
    bool parsed;
    /// metadata for state
    bool connected;
    /// metadata for state
    bool converged;

    /// log file stream
    ostream& outStream;

    /// constructor (when data is provided as a 9-char string)
    State(string data, map<string, State*> *m,ostream& os): outStream(os){
        dataString = std::move(data);
        stateMap = m;

        status = check(dataString);
        // status = '-';
        if(status == '-'){
            terminal = false;
            parsed = false;
            connected = false;
            converged = false;
            optimalAction = -1;
            value = 0;
        }
        else{
            terminal = true;
            parsed = true;
            connected = true;
            converged = true;
            optimalAction = -1;

            // if state is terminal, prescribe the values accordingly.
            if     (status == 'O'){value = 1  ;}
            else if(status == 'X'){value = 0  ;}
            else if(status == 'D'){value = 0.5;}
            else{os << "[STATE-CONSTRUCTOR] Wrong Status!";}
        }
    }

    /// constructor (when data is provided as Eigen::Matrix3d) (delegated)
    State(const Matrix3d& data, map<string, State*> *m,ostream& os): State(tttString(data),m,os){}

    bool operator== (const State &s) const{return (dataString == s.dataString);}
    bool operator== (const string &m) const{return (dataString == m);}
    friend ostream& operator << (ostream& os, const State& s);
    friend ostream& operator << (ostream& os, vector<int>& P);

    /// verbose output for full data analysis
    void verbose(ostream& os){
        os << "["<<this<<"]:\n" << tttShow(dataString,'\n') <<endl;
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
        if(converged){os << "Optimal Action: " << optimalAction << " (value: "<< value <<")"<< endl;}
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
        tDataString[action] = 'O';

        outStream << "[PARSE-TRANS "<<dataString <<"-"<<action<<"] (-> "<< tDataString << ")" << endl;

        // The case when Agent wins or Draws immediately
        if(check(tDataString) == 'O'){
            outStream << dataString << "> a:" << action <<" (W) -> " << tDataString <<" | ";
            transS[idx].push_back(getStatePtr(tDataString));
            transP[idx].push_back(1);
            // transR[idx].push_back(1); // wrong implementation
            transR[idx].push_back(0);
            return;
        }
        else if(check(tDataString) == 'D'){
            outStream << dataString << "> a:" << action <<" (D) -> "<< tDataString << " | ";
            transS[idx].push_back(getStatePtr(tDataString));
            transP[idx].push_back(1);
            // transR[idx].push_back(0.5); //wrong implementation.
            transR[idx].push_back(0);
            return;
        }

        // The case when opponent gets to make a move
        oActions = actionsFrom(tDataString);
        outStream << "Possible Opponent Actions: "<< oActions << endl;

        string nDataString;
        for(auto a:oActions){
            nDataString = tDataString;
            nDataString[a] = 'X';
            outStream << dataString << "> a:" << action << "/o:" << a << " -> " << nDataString << " | ";
            transS[idx].push_back(getStatePtr(nDataString));
            transP[idx].push_back(1/double(oActions.size()));
            if(check(nDataString) == 'D'){
                // transR[idx].push_back(0.5);
                transR[idx].push_back(0);
            }
            else{transR[idx].push_back(0);}
        }
    }

    /// parse the data within the state (populate all data)
    void parse(){
        if(parsed){return;}
        parseActions();
        for(int idx = 0;idx < actions.size();idx++){
            parseTrans(idx);
        }
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
            outStream << "[STATE " << dataString << "] Created: " << *sp << endl;
        }
        else{
            sp = stateMap -> find(newDataString) -> second;
            outStream << "[STATE " << dataString << "] Linked : " << *sp << endl;
        }
        return sp;
    }

    /// calculates the value function (only if "downstream" states are calculated)
    bool calcValue(){

        /// return true if value is already calculated
        if(converged){return true;}

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

        /// calculate state-action values for each actions[idx]
        for(int idx = 0;idx < actions.size();idx++){
            double temp = 0;
            /// the VI formula is implemented here
            for(int idxx = 0; idxx < transR[idx].size();idxx++){
                temp += transP[idx][idxx]*(transR[idx][idxx] + GAMMA*(transS[idx][idxx] -> value));
            }
            values.push_back(temp);
        }

        /// choose max state-action value as value
        value = *max_element(values.begin(),values.end());

        /// choose corresponding optimal action
        optimalAction = actions[int(max_element(values.begin(),values.end()) - values.begin())];
        // outStream << "optimal action of " << dataString << ":" << optimalAction << endl;
        converged = true;
        return true;
    }

};

/// overloaded << operator for concise output of State class varables.
ostream & operator<< (ostream& os, const State& s) {
    os << "<" << &s << "> ";
    os << tttShow(s.dataString,'|');
    os << " {" << s.status << "}";
    os << " (" << s.value << ")";

    if(s.terminal){os << " (T)";}
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

/// Loop to create / populate the states needed for VI
State* stateLoop(const Matrix3d& startStateMat,map<string, State*> *stateMap,ostream& os){
    auto startState = new State(startStateMat,stateMap,os);
    stateMap -> insert({startState->dataString,startState});

    os << "[SL] Starting State: " << *startState << endl;

    startState -> connect();

    // startState -> verbose(os);

    os << "[SL-FIN] All relevant states created (size: "<<stateMap -> size()<<"): "<<endl;
    // os << "[SL-FIN] States created. Showing stateMap (size:" << stateMap -> size() << "):" <<endl;
    // for (auto const &pair: *stateMap) {
    //     os << "\t{" << pair.first << ": " << *pair.second << "}\n";
    // }

    //os << startState -> stateMap << endl;
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
            if(e.second -> calcValue()){
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
double getOptimalValue(Matrix3d state){

    auto start = chrono::high_resolution_clock::now();
    auto now = start;

    ofstream ofs("./log.txt");
    ostream& os = ofs;

    map<string, State*> stateMap;
    auto s = stateLoop(state,&stateMap,os);

    auto slDuration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - now);
    now = chrono::high_resolution_clock::now();
    os << "[STATE-LOOP] Finished (elapsed time: " << slDuration.count() <<"us)" << endl;

    valueIterLoop(s,&stateMap,os);

    auto viDuration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - now);
    os << "[VALUE-ITER] Finished (elapsed time: " << viDuration.count() <<"us)" << endl;

    os << "Final Result (state analysis):"<< endl;
    s ->verbose(os);

    auto totDuration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
    os << "Total Elapsed Time: " << totDuration.count() << "us" << endl;

    return s->value; // return optimal value
}

/// simple enum to input Agent moves / Opponent moves / empty slots
// enum TTT{
//     O = 1,
//     X = -1,
//     _ = 0,
// };
// int main() {
//     Eigen::Matrix3d state;
//     state<<
//     O, _, _,
//     _, X, _,
//     _, _, _
//
//     cout<<"optimal value for state: "<<getOptimalValue(state)<<std::endl;
//
//     return 0;
// }
