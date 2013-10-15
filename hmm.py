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

def getTrainSeqs(train_file):
    '''get word dict and  observation sequence and tag sequence from training set'''

    print getTrainSeqs.__doc__

    print 'read training set and build word dict with no space...'
    lines=[l.decode('utf8').strip() for l in train_file]
    wordset=set()
    wordmap={}
    tags=[]

    for line in lines:
        for w in removeSpace(line):
            wordset.add(w)

    i=0

    for w in wordset:
        wordmap[w]=i
        i=i+1

    print 'training set have %d lines and %d distinct words'%(len(lines),i)
    print 'begin to get observation sequence and tag sequence of training set...'
    tick=0.05
    current=0
    total=len(lines)
    nprocessed=0

    seqs=[]
    tseqs=[]
    for line in lines:
        nprocessed+=1
        if nprocessed>current*tick*total:
            print 'now %d%% of training set have been processed'%(current*tick*100)
            current+=1
        seq=[]
        tseq=[]
        wseq=[]
        lastspace=True
        for i in range(len(line)):
            if not isSpace(line[i]):
                seq.append(wordmap[line[i]])
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
        seqs.append(seq)
        tseqs.append(tseq)

    return (wordmap,seqs,tseqs)

def createHmm(wordmap,seqs,tseqs):
    '''create hmm '''

    print createHmm.__doc__

    observations=[]
    states=[]
    for seq in seqs:
        observations.extend(seq)
    for tseq in tseqs:
        states.extend(tseq)

    nstates=4
    nobservations=6000 #len(wordmap)+1

    Pi=[0.5,0.5,0,0]
    A=[[0.0]*nstates for j in range(nstates)] #tranlate propobility matrix
    B=[[1.0]*nobservations for j in range(nstates)] #observation distribution matrix

    C=[0.0]*nstates  #count of each state
    C[states[0]]=1 

    
    for i in range(1,len(states)):
        s1=states[i-1]
        s2=states[i]
        A[s1][s2]+=1
        C[s2]+=1
        B[s2][int(observations[i])]+=1

    for i in range(nstates):
        for j in range(nstates):
            A[i][j]=A[i][j]/C[i]
    for i in range(nstates):
        for j in range(nobservations):
            B[i][j]=B[i][j]/C[i]

    return (nstates,nobservations,A,B,Pi)

def saveTrainSeqs(wordmap,dict_file,seqs,id_seq_file,tseqs,tag_seq_file):
    '''save wordmap,seqs,tseqs,wseqs to file'''
   
    if dict_file:
        print 'write word dict to file...'
        import operator
        words=[unicode(k+' '+str(v)) for (k,v) in sorted(wordmap.iteritems(), key=operator.itemgetter(1))]
        dict_file.write('\r\n'.join(words).encode('utf8'))
        dict_file.close()

    if id_seq_file:
        print 'write integer id sequences to file...'
        seqs=[[str(id) for id in seq] for seq in seqs]
        seqs=[' '.join(seq) for seq in seqs]
        id_seq_file.write('\r\n'.join(seqs))
        id_seq_file.close()

    if tag_seq_file:
        print 'write integer tag sequence to file...'
        tseqs=[[str(tag) for tag in tseq] for tseq in tseqs]
        tseqs=[' '.join(tseq) for tseq in tseqs]
        tag_seq_file.write('\r\n'.join(tseqs))
        tag_seq_file.close()

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
    dict_file=file(dict_file_name,'w')

    (wordmap,seqs,tseqs)=getTrainSeqs(train_file)
    hmm=createHmm(wordmap,seqs,tseqs)
    saveHmm(hmm_file,hmm)
    saveTrainSeqs(wordmap,dict_file,None,None,None,None)

def getTestSeqs(dict_file,test_file):
    '''convert test set to [integer observation sequence] and [word sequence]
    '''
    print getTestSeqs.__doc__

    tmp=[l.decode('utf8').strip().split()for l in dict_file]
    tmp=[(l[0],int(l[1])) for l in tmp]
    wordmap=dict(tmp)

    unknown_id=len(wordmap)

    lines=[line.decode('utf8').strip() for line in test_file]
    tick=0.05
    current=0
    total=len(lines)
    nprocessed=0

    print 'test set have %d lines'%len(lines)
    print 'begin to get observation sequence from test set...'
    seqs=[]
    wseqs=[]
    for line in lines:
        nprocessed+=1
        if nprocessed>current*tick*total:
            print 'now %d%% have been processed'%(current*tick*100)
            current+=1

        seq=[]
        wseq=[]
        for w in removeSpace(line):
            wseq.append(w)
            seq.append(wordmap[w] if wordmap.has_key(w) else unknown_id)
        if seq:
            seqs.append(seq)
            wseqs.append(wseq)
    
    return (seqs,wseqs)

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

def getStateSeqs(hmm,o_seqs):
    '''create state sequence  from observation sequence with hmm'''
   
    print getStateSeqs.__doc__

    (_,_,A,B,Pi)=hmm
    
    tick=0.05
    current=0
    total=len(o_seqs)
    nprocessed=0

    print 'begin to get state sequence of test set...total:%d'%total
    s_seqs=[]
    for o_seq in o_seqs:
        nprocessed+=1
        if nprocessed>current*tick*total:
            print 'now %d%% have been processed'%(current*tick*100)
            current+=1
        s_seq=viterbi(A,B,Pi,o_seq)
        s_seqs.append(s_seq)

    return s_seqs

def saveResult(result_file,tseqs,wseqs):
    '''save result into file from tag sequences and word sequence '''

    print saveResult.__doc__

    seg_lines=[]
    for i in range(len(tseqs)):
        tseq=tseqs[i]
        wseq=wseqs[i]
        seg_line=[]
        for j in range(len(tseq)):
            if tseq[j]==0 or tseq[j]==2:
                seg_line.append(wseq[j]+' ')
            else:
                seg_line.append(wseq[j])

        seg_lines.append(''.join(seg_line))

    result_file.write('\r\n'.join(seg_lines).encode('utf8'))
    result_file.close()

def segment(dict_file_name='dict.txt',hmm_file_name='model.hmm',test_file_name='msr_test.utf8',result_file_name='test_result.utf8'):

    dict_file=file(dict_file_name)
    test_file=file(test_file_name)
    result_file=file(result_file_name,'w')

    hmm=parseHmm(file(hmm_file_name))

    (o_seqs,wseqs)=getTestSeqs(dict_file,test_file)
    s_seqs=getStateSeqs(hmm,o_seqs)

    saveResult(result_file,s_seqs,wseqs)

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
