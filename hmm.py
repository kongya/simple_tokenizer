#!/usr/bin/env python
#encoding=utf8
import sys,getopt
import re

w_space_pattern=re.compile(ur'^\s+$')
l_space_pattern=re.compile(ur'\s+')

def isSpace(word):
    return True if w_space_pattern.match(word) else False
def removeSpace(line):
    return l_space_pattern.sub('',line)
def getWordMap(dict_file):
    '''build word map from dict file:first line should be <\s> 0 '''
    tmp=[l.decode('utf8').strip().split()for l in dict_file]
    wordmap=dict([(l[0],int(l[1])) for l in tmp if len(l)==2])
    return wordmap

def getTrainSeqs(train_file,wordmap):
    '''get word dict and  observation sequence and tag sequence from training set'''

    print getTrainSeqs.__doc__

    
    print 'begin to get observation sequence and tag sequence of training set...'
    nprocessed=0
    unknown_id = len(wordmap)
    for line in train_file:
        line=line.decode('utf8').strip()
        nprocessed+=1
        if nprocessed%100==0:
            print 'now %d lines of training set have been processed'%(nprocessed)
          
        seq=[]
        tseq=[]
        lastspace=True
        for i in range(len(line)):
            if not isSpace(line[i]):
                if wordmap.has_key(line[i]):
                    seq.append(wordmap[line[i]])
                else:
                    seq.append(unknown_id)
                if(lastspace):
                    if(i==len(line)-1 or line[i+1]==' '):
                        tseq.append(0) #S
                    else:
                        tseq.append(1)  #B
                else:
                    if(i==len(line)-1 or line[i+1]==' '):
                        tseq.append(2) #E
                    else:
                        tseq.append(3) #M
                lastspace=False
            else:
                lastspace=True

        if seq:
            yield (seq,tseq)


def createHmm(wordmap,pairs):
    '''create hmm '''

    print createHmm.__doc__

    nstates=4
    nobservations=len(wordmap)+200

    Pi=[0]*nstates
    A=[[0.0]*nstates for j in range(nstates)] #tranlate propobility matrix
    B=[[1.0]*nobservations for j in range(nstates)] #observation distribution matrix
    C=[0.0]*nstates  #count of each state

    nseqs=0
    for (seq,tseq) in pairs:
        Pi[tseq[0]]+=1
        nseqs+=1
        C[tseq[0]]+=1 
        for i in range(1,len(tseq)):
            s1=tseq[i-1]
            s2=tseq[i]
            A[s1][s2]+=1
            C[s2]+=1
            B[s2][int(seq[i])]+=1

    Pi=[i/float(nseqs) for i in Pi]
  
    for i in range(nstates):
        for j in range(nstates):
            A[i][j]=A[i][j]/C[i]
    for i in range(nstates):
        for j in range(nobservations):
            B[i][j]=B[i][j]/C[i]

    return (nstates,nobservations,A,B,Pi)


def saveHmm(hmm_file,hmm):
    '''save hmm to file'''

    print saveHmm.__doc__

    (nstates,nobservations,A,B,Pi)=hmm

    header='\r\n'.join(['Hmm','v1.0','NbStates '+str(nstates),'\r\n'])
    hmm_file.write(header)

    for i in range(nstates):
        state='\r\n'.join(['State','Pi %.15f'%Pi[i],'A '+' '.join(['%.15f'%Aij for Aij in A[i]]),'IntegerOPDF ['+' '.join(['%.15f'%Bij for Bij in B[i]])+']','\r\n'])
        hmm_file.write(state)

    hmm_file.close()


def train(train_file_name='msr_training.utf8',hmm_file_name='model.hmm',dict_file_name='dict.txt'):
    train_file=file(train_file_name)
    hmm_file=file(hmm_file_name,'w')
    dict_file=file(dict_file_name)
    wordmap=getWordMap(dict_file)

    pairs=getTrainSeqs(train_file,wordmap)
    hmm=createHmm(wordmap,pairs)
    saveHmm(hmm_file,hmm)

def getTestSeqs(dict_file,test_file):
    '''convert test set to [integer observation sequence] and [word sequence]
    '''
    print getTestSeqs.__doc__

    wordmap=getWordMap(dict_file)

    unknown_id=len(wordmap)

    nprocessed=0
    print 'begin to get observation sequence from test set...'

    for line in test_file:
        line=line.decode('utf8').strip() 
            
        seq=[]
        wseq=[]
        for w in l_space_pattern.sub(',',line):
            wseq.append(w)
            seq.append(wordmap[w] if wordmap.has_key(w) else unknown_id)
        if seq:
            yield (seq,wseq)


from math import log

def _max(l):
    if not l:
        return (-1,None)
    mv=l[0]
    mi=0

    for i in range(1,len(l)):
        if l[i]>mv:
            mv=l[i]
            mi=i
    
    return (mi,mv)    

def viterbi(A,B,Pi,O):
    N=len(Pi)  # # of states
    T=len(O)   # len of observation sequence

    delta=[-sys.maxint]*N
    for j in range(N):
        if Pi[j]!=0 and B[j][O[0]]!=0:
            delta[j]=log(Pi[j])+log(B[j][O[0]])
    
    psy=[[0]*N for t in range(T)]
    states=[0]*T

    for t in range(1,T):
        Ot=O[t]
        delta_new=[0]*N
        for j in range(N):
            tmp=[delta[i]+log(A[i][j]) if A[i][j]!=0 else -sys.maxint for i in range(N) ]
            (mi,mv)=_max(tmp)
            tmp=log(B[j][Ot]) if B[j][Ot]!=0 else -sys.maxint
            delta_new[j]=mv+tmp
            psy[t][j]=mi

        delta=delta_new

    (states[T-1],_)=_max(delta)
    for t in range(T-2,-1,-1):
        states[t]=psy[t+1][states[t+1]]

    return states
class FileFormatException(Exception):
    pass

def parseHmm(hmm_file):
    lines=[l.decode('utf8').strip() for l in hmm_file]
    i=0
    last=-1

    NAME=0
    VERSION=1
    NBSTATES=2
    STATE=3
    SPI=4
    SA=5
    SB=6

    nstates=0
    nobservations=-1
    A=[]
    B=[]
    Pi=[]

    for l in lines:
        i+=1
        if len(l)==0 or l[0]=='#':
            continue
        else:
            if last==-1:
                if l!='Hmm':
                    raise FileFormatException('line #%d: Hmm expected\r\n'%i)
                else:
                    last=NAME
                    continue
            if last==NAME:
                if l!='v1.0':
                    raise FileFormatException('line #%d: v1.0 expected\r\n'%i)
                else:
                    last=VERSION
                    continue
            if last==VERSION:
                if not l.startswith('NbStates '):
                    raise FileFormatException('line #%d: NbStates expected\r\n'%i)
                else:
                    last=NBSTATES
                    nstates=int(l.split()[1])
                    continue
            if last==NBSTATES or last==SB:
                if l!='State':
                    raise FileFormatException('line #%d: State expected\r\n'%i)
                else:
                    last=STATE
                    continue
            if last==STATE:
                if not l.startswith('Pi '):
                    raise FileFormatException('line #%d: Pi expected\r\n'%i)
                else:
                    last=SPI
                    Pi.append(float(l.split()[1]))
                    continue
            if last==SPI:
                if not l.startswith('A '):
                    raise FileFormatException('line #%d: A expected\r\n'%i)
                else:
                    last=SA
                    A.append([float(f) for f in l.split()[1:]])
                    continue
            if last==SA:
                if not l.startswith('IntegerOPDF '):
                    raise FileFormatException('line #%d: IntegerOPDF expected\r\n'%i)
                else:
                    last=SB
                    l=l[len('IntegerOPDF'):].strip()
                    if len(l)==0 or l[0]!='[':
                        raise FileFormatException('line #%d: [ expected\r\n'%i)
                    if l[-1]!=']':
                        raise FileFormatException('line #%d: ] expected\r\n'%i)
                    l=l[1:-1]
                    Bi=[float(f) for f in l.split()]
                    if nobservations==-1:
                        nobservations=len(Bi)
                    elif nobservations!=len(Bi):
                        raise FileFormatException('line #%d: # of items should be %d\r\n'%(i,nobservations))
                    B.append([float(f) for f in l.split()])
                    continue
    if last!=SB:
        raise FileFormatException('line #%d: not complete\r\n'%i)

    return (nstates,nobservations,A,B,Pi)

def getStateSeqs(hmm,seq_wseq_pairs):
    '''create state sequence  from observation sequence with hmm'''
   
    print getStateSeqs.__doc__

    (_,_,A,B,Pi)=hmm

    print 'begin to get state sequence of test set...'
    for (o_seq,w_seq) in seq_wseq_pairs:
        s_seq=viterbi(A,B,Pi,o_seq)
        yield (s_seq,w_seq)

def saveResult(result_file,tseq_wseq_pairs):
    '''save result into file from tag sequences and word sequence '''

    print saveResult.__doc__

    seg_lines=[]
    nprocessed=0
    for (tseq,wseq) in tseq_wseq_pairs:
        seg_line=[]
        if nprocessed%100==0:
            print 'now %d lines have been processed'%nprocessed
        nprocessed+=1
        
        for j in range(len(tseq)):
            if tseq[j]==0 or tseq[j]==2:
                seg_line.append(wseq[j]+' ')
            else:
                seg_line.append(wseq[j])
        line=''.join(seg_line)+'\r\n'
        result_file.write(line.encode('utf8'))

    result_file.close()
    

def segment(dict_file_name='dict.txt',hmm_file_name='model.hmm',test_file_name='msr_test.utf8',result_file_name='test_result.utf8'):

    dict_file=file(dict_file_name)
    test_file=file(test_file_name)
    result_file=file(result_file_name,'w')

    hmm=parseHmm(file(hmm_file_name))

    seq_wseq_pairs=getTestSeqs(dict_file,test_file)
    tseq_wseq_pairs=getStateSeqs(hmm,seq_wseq_pairs)

    saveResult(result_file,tseq_wseq_pairs)

def usage():
    print '''usage:
        python hmm.py -t train_file result_model_file dict_file
        or python hmm.py -s dict_file model_file  test_file result_segmented_file
    '''
    
def main(argv):
    if len(argv)==4 and argv[0]=='-t':
        train(argv[1],argv[2],argv[3])
    elif len(argv)==5 and argv[0]=='-s':
        segment(argv[1],argv[2],argv[3],argv[4])
    else:
        usage()
        
if __name__ == '__main__':
    main(sys.argv[1:])
