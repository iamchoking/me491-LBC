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
ostream& operator << (ostream& os, vector<vector<int>>& P);

ostream& operator << (ostream& os, vector<double>& P);

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


//  -- --    0  1
// |  |  |  6  8  10
//  -- --    2   3
// |  |  |  7  9  11
//  -- --    4   5

/// internal helper function to make dab-strings more readable
string dabShow(const string& dabString, const char& delimiter,int tabL = 0){
    string l1 = " -- -- ";
    string l2 = "|  |  |";
    string l3 = " -- -- ";
    string l4 = "|  |  |";
    string l5 = " -- -- ";

    if(delimiter == '\n'){
        l1 += "\t   0  1   ";
        l2 += "\t  6  8  10";
        l3 += "\t   2   3  ";
        l4 += "\t  7  9  11";
        l5 += "\t   4   5  ";
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

    l1 = string(tabL,'\t') + l1;

    if(delimiter == '\n'){
        l2 = string(tabL,'\t') + l2;
        l3 = string(tabL,'\t') + l3;
        l4 = string(tabL,'\t') + l4;
        l5 = string(tabL,'\t') + l5;
    }

    return(l1+delimiter+l2+delimiter+l3+delimiter+l4+delimiter+l5);
}

/// counts how many "boxes" are already closed in this state.
int countClosed(const string& dab){
    int numClosed = 0;
    if(dab[0] == '-' && dab[2] == '-' && dab[6] == '-' && dab[8 ] == '-'){numClosed += 1;}
    if(dab[1] == '-' && dab[3] == '-' && dab[8] == '-' && dab[10] == '-'){numClosed += 1;}
    if(dab[2] == '-' && dab[4] == '-' && dab[7] == '-' && dab[9 ] == '-'){numClosed += 1;}
    if(dab[3] == '-' && dab[5] == '-' && dab[9] == '-' && dab[11] == '-'){numClosed += 1;}
    return numClosed;
}

int countEmpty(const string& dab){
    int empty = 0;
    for(auto c:dab){if(c=='_'){empty++;}}
    return empty;
}

/// checks the status of board given the string (true: terminal, false: non-terminal)
bool check(const string& dab){
    return dab == "------------";
}

/// extracting possible actions from a given string (return: a vector of indices of '_')
vector<vector<int>> actionsFrom(const string& dab,vector<int> prefix = vector<int>()){
    vector<vector<int>> actions;
    int closed = countClosed(dab);

    int idx;
    string temp_string = dab;
    for(idx = 0; idx < dab.length();idx++) {
        if (dab[idx] == '_') {
            vector<int> a;
            temp_string = dab;
            temp_string[idx] = '-';
            if(countClosed(temp_string) > closed && !check(temp_string)){ //a box was closed
                auto nprefix = vector<int>();
                copy(prefix.begin(),prefix.end(), back_inserter(nprefix));
                nprefix.push_back(idx);
                auto newActions = actionsFrom(temp_string,nprefix); //new set of actions with prefix
                actions.insert(actions.end(),newActions.begin(),newActions.end());
            }
            else{
                //vector1.insert( vector1.end(), vector2.begin(), vector2.end() );
                //actions.push_back(vector<int>());
                actions.emplace_back();
                copy(prefix.begin(),prefix.end(), back_inserter(actions.back()));
                actions.back().push_back(idx);
            }
        }
    }
    return actions;
}

void takeAction(string& dab,const vector<int>& a){
    for(int idx:a){dab[idx] = '-';}
}

class State{
public:
    /// the string of length 12 that holds the board info.
    string dataString;
    // the number of "boxes" that are already closed
    int numClosed;
    /// pointer to hash table of State objects
    map<string, State*> *stateMap;

    /// vector of possible action sqeuences (action "{a,b,c}" means putting "1" sqeuentially on index a,b,c)
    vector<vector<int>> actions;

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
        numClosed = countClosed(dataString); // later used to determine rewards

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
        os << "Showing:" << *this << endl;
        os << dabShow(dataString,'\n') <<endl;
        if(terminal){
            os<< "[TERMINAL STATE]"<<endl;
            return;
        }
        os << "Actions: " << actions;
        os << endl;
        os << "Transitions: " << endl;
        for(int idx = 0;idx < actions.size();idx++){
            os << "\tAction[" << idx << "]: " << actions[idx];
            double p = 0;
            for(auto a:transP[idx]){p += a;}
            os << "  >> "<< size(transS[idx]) << " Possible Transitions (probability sums to " << p <<")";
            if(idx == policy){os << " <<< [CHOSEN BY POLICY]";}
            os<<endl;
            for(int idxx = 0;idxx<transS[idx].size();idxx++){
                os << "\t\t" << *transS[idx][idxx] << endl;
                os << dabShow(transS[idx][idxx]->dataString,'\n',2) << endl;
                os << "\t\t" << " (p: "<<transP[idx][idxx]<<" r: " << transR[idx][idxx] << " v: " << transS[idx][idxx] -> v << ")" << endl << endl;
            }
        }
        if(converged){os << "Current policy: " << actions[policy] << " (value: "<< v <<")"<< endl;}
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
    void parseTrans(int idx,bool verbo = true){
        auto action = actions[idx];

        vector<vector<int>> oActions;
        vector<string> nDataStrings;

        string tDataString = dataString;
        takeAction(tDataString,action);
        //tDataString[action] = '-'; // DONE: implement placing actions

        // reward is the difference in closed boxes
        //reward is determined as soon as action takes place
        auto reward = double(countClosed(tDataString)-numClosed);

        if(verbo){outStream << "[PARSE-TRANS "<<dataString <<"&"<<action<<"] (-> "<< tDataString << ")" << endl;}

        // The case when Agent finishes the game immediately
        if(check(tDataString)){ //is terminal
            if(verbo){outStream << dataString << " > a:" << action <<" (T) -> " << tDataString <<" | ";}
            transS[idx].push_back(getStatePtr(tDataString,verbo));
            transP[idx].push_back(1);
            transR[idx].push_back(reward);
            return;
        }

        // The case when opponent gets to make a move
        oActions = actionsFrom(tDataString);
        if(verbo){outStream << "Possible Opponent Actions: "<< oActions << endl;}

        string nDataString;
        const int empty = countEmpty(tDataString);

        for(auto a:oActions){
            nDataString = tDataString;
            takeAction(nDataString,a);
            //nDataString[a] = '-'; DONE: implement action placement
            if(verbo){outStream << dataString << " > a:" << action << "/of:" << a << " -> " << nDataString << " | ";}
            transS[idx].push_back(getStatePtr(nDataString,verbo));

            //(TRICKY) add proper probability to each opponent action
            double de = 1; //denominator for probability
            for(int t = 0;t<size(a);t++){de = de*(empty-t);}
            transP[idx].push_back(1.00/de);
            transR[idx].push_back(reward); // reward is the difference in closed boxes
        }

    }

    /// parse the data within the state (populate all data)
    void parse(bool verbo = true){
        if(parsed){
            //outStream << "[PARSE-SKIP] Skipped" << endl;
            return;
        }
        outStream << "[PARSE] Parsing State ["<<dataString<<"]";
        if(verbo){outStream << endl;}

        parseActions();
        for(int idx = 0;idx < actions.size();idx++){parseTrans(idx,verbo);}

        policy = int(rand()) % size(actions); //initialize with a random policy
        outStream << " ++ Policy Initialized To [" << actions[policy] << "](idx: "<<policy <<")"<<endl;

        parsed = true;

    }

    /// recursively create and parse all data what is "downstream" from current state
    bool connect(bool verbo){
        parse(verbo);
        if(connected){return false;}
        for(const auto& tr:transS){
            for(auto s:tr){
                if(s -> connect(verbo)){}
            }
        }
        if(verbo){outStream << "[CONNECT]" << "Connection of [" << dataString << "] Complete." << endl;}
        connected = true;
        return true;
    }

    /// reference the hash table to return the State that represents the given data string, or create one if needed.
    [[nodiscard]] State* getStatePtr(const string& newDataString,bool verbo) const{
        State* sp;
        auto tmp = stateMap ->find(newDataString);

        if(stateMap -> find(newDataString) == stateMap -> end()){
            sp = new State(newDataString,stateMap,outStream);
            stateMap -> insert({newDataString,sp});
            //outStream << "[STATE " << dataString << "] Created: " << *sp << endl;
            if(verbo){outStream << "[STATE " << dataString << "] Created: [" << sp << "]" << newDataString << endl;}
        }
        else{
            sp = stateMap -> find(newDataString) -> second;
            //outStream << "[STATE " << dataString << "] Linked : " << *sp << endl;
            if(verbo){outStream << "[STATE " << dataString << "] Linked : [" << sp << "]" << newDataString << endl;}
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

        //outStream << "[VI] for " << dataString << "on <"<<actions[policy]<< ">: P:" << transP[policy] << "| R:" << transR[policy]<< "| v': ";
        /// the VI formula is implemented here
        for(int idxx = 0; idxx < size(transR[policy]);idxx++){
            temp += transP[policy][idxx] * (transR[policy][idxx] + GAMMA*(transS[policy][idxx] -> v) );
            //outStream << transS[policy][idxx] -> v << ", ";
        }
        //outStream << endl;
        v = temp;
        //outStream << "[VI] Result: " << this -> v<<endl;

        converged = true;
        return true;
    }

    bool improveState(){
        if(!converged || terminal){ // all improvements need to operate on converged v
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
        int oldP = policy;
        policy = int(distance(q.begin(), max_element(q.begin(),q.end())));

        // initialize setup for subsequent evaluation.
        converged = terminal;
        return oldP != policy; // returns true if policy has changed
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
    os << "<";
    for (int i : P)
        os << i << "," ;
    os << ">";
    return os;
}

ostream& operator << (ostream& os, vector<vector<int>>& P) {
    os << "<";
    for (auto i : P)
        os << i << "," ;
    os << ">";
    return os;
}

ostream& operator << (ostream& os, vector<double>& P) {
    for (double i : P)
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
State* stateLoop(const Eigen::Vector<int, 12>& startStateVector,map<string, State*> *stateMap,ostream& os,bool verbose = false){

    auto startState = new State(startStateVector,stateMap,os);

    stateMap -> insert({startState->dataString,startState});

    os << "[SL] Starting State: " << *startState << endl;

    startState -> connect(verbose);
    //startState -> connect(true);

    // startState -> verbose(os);

    os << "[SL-FIN] All relevant states created (size: "<<stateMap -> size()<<"): "<<endl;
    // os << "[SL-FIN] States created. Showing stateMap (size:" << stateMap -> size() << "):" <<endl;

    //os << *(startState -> stateMap) << endl;
    return startState;
}

/// VI loop: iterates over all states in stateMap until all of them converge (usually, it takes only one iteration!)
void policyEvalLoop(State* s, const map<string, State*> *stateMap, ostream& os){
    int numStates = int(stateMap -> size());
    os << "\t[EVAL-START] Value Iteration for " << numStates << " States :" << endl;

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
        os << "\t[EVAL-ITER" << iter << "] " << numConverged << " out of " << numStates << " converged" << endl;
    }

    //os << "[EVAL-FIN] final statemap (size "<< stateMap -> size() << " ): "<<endl;
    //os << *stateMap << endl;
}

int policyImprLoop(State* s, const map<string, State*> *stateMap, ostream& os){
    int numStates = int(stateMap -> size());
    os << "\t[IMPROVE] Policy Improvement for " << numStates << " States :" << endl;
    int numChanged = 0;

    for(auto e: *stateMap){
        // os << "checking: [" << e.first <<":"<< stateMap.find(e.first) -> second << "]" << endl;
        // if(e.second == 0){continue;}
        if(e.second -> improveState()){
            numChanged += 1;
        }
    }
    os << "\t[IMPROVE] " << numChanged << " states (out of " << numStates << ") changed policy" << endl;
    return numChanged;
}

// DO NOT CHANGE THE NAME AND FORMAT OF THIS FUNCTION
/// final outer loop for VI workflow (executes stateLoop -> policyEvalLoop & records elapsed time)
State* policyIteration(const Eigen::Vector<int, 12>& state,const string& logPath = "./log.txt"){

    auto start = chrono::high_resolution_clock::now();
    auto now = start;
    ofstream ofs(logPath);
    ostream& os = ofs;
    //ostream& os = cout;

    map<string, State*> stateMap;

    auto s = stateLoop(state,&stateMap,os);

    auto slDuration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - now);
    now = chrono::high_resolution_clock::now();

    os << "[PI] stateLoop Finished (elapsed time: " << slDuration.count() <<"us)" << endl;
    //s -> verbose(os);

    int iter = 0;
    int updates = -1;
    while (updates != 0) {
        os << "[PI-" << iter << "] Started" << endl;

        policyEvalLoop(s, &stateMap, os);
        auto peDuration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - now);
        now = chrono::high_resolution_clock::now();
        os << "\t[PI-" << iter << "] Policy Evaluation Finished (elapsed time: " << peDuration.count() << "us)"
           << endl;
        os << "\t[PI-VALUE] Current value for starting state: " << s->v << endl;

        os << endl;

        updates = policyImprLoop(s, &stateMap, os);
        auto piDuration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - now);
        now = chrono::high_resolution_clock::now();
        os << "\t[PI-" << iter << "] Policy Improvement Finished (elapsed time: " << piDuration.count() << "us)"<< endl;

        os << endl;
        iter++;
    }

    os << "Final Result (state analysis):"<< endl;
    s ->verbose(os);


    os << "[PI] Finished (no policy updates) in " << iter << " iterations" << endl;
    if(! s -> terminal){os << "[Optimal Action: " << s -> actions[s -> policy] << ", Value: " << s -> v << "]"<< endl;}

    auto totDuration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
    os << "Total Elapsed Time: " << totDuration.count() << "us" << endl;

    return s; // return optimal value
}


// SKELETON: HW2
/// DO NOT CHANGE THE NAME AND FORMAT OF THIS FUNCTION
double getOptimalValue(const Eigen::Vector<int, 12>& state){
    // return the optimal value given the state
    auto path = "./log_[" + dabString(state) + "]-V.txt";
    return policyIteration(state,path) -> v;
}

/// DO NOT CHANGE THE NAME AND FORMAT OF THIS FUNCTION
int getOptimalAction(const Eigen::Vector<int, 12>& state){
    // return one of the optimal actions given the state.
    // the action should be represented as a state index, at which a line will be drawn.
    auto path = "./log_[" + dabString(state) + "]-A.txt";
    auto s = policyIteration(state,path);
    auto policyIdx = s -> policy;

    if(s -> terminal){return -1;}
    return s -> actions[policyIdx][0];  // return optimal action
}

//==================20190673.hpp=================

//========== SKELETON CODE ==========
int main() {
    Eigen::Vector<int, 12> state;
    //state << 0,0,0,0,0,0,0,0,0,0,0,0; // 3.97701
    //state << 1,0,0,0,0,0,0,0,0,0,0,0; // 3.51715
    //state << 1,1,0,0,0,0,0,0,0,0,0,0; // 3.98413

    state << 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0; // 3.2 ??
        //state << 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0; // 3.2 ??

    //state << 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0; // 3.33333
    //state << 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1; // 2.33333
    //state << 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0; // 4
    //state << 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0; // 3.33333

    //Sanity Checks
    //state << 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0; // must be 4
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
