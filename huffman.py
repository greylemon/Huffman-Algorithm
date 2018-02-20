import doctest, time
from nodes import HuffmanNode, ReadNode


    # ====================
    # Priority_Queue class # helper for huffman_tree function

class Priority_Queue:
    '''
    Priority Queue ADT
    A set of items stored, with each item having a priority
    '''
    def __init__(self: 'Priority_Queue') -> None:
        '''Initialize Priorit y_Queue
        >>> PQ = Priority_Queue()
        >>> type(PQ) == Priority_Queue
        True
        '''
        self.size = 0                          
        self._items = []                          

    def enqueue(self: 'Priority_Queue', priority: int, item: object) -> None:
        '''
        Adds an item to the priority queue
        >>> PQ = Priority_Queue()
        >>> PQ.is_empty()
        True
        >>> PQ.enqueue(1,2)
        >>> PQ.is_empty()
        False
        '''
        self.size = self.size + 1                   
        self._items.append([priority,item])        
    
    def dequeue(self: 'Priority_Queue') -> object:
        '''
        Precondition: there is at least one element in priority queue
        Removes and returns an item with the highest priority

        >>> PQ = Priority_Queue()
        >>> PQ.enqueue(0,5)
        >>> PQ.is_empty()
        False
        >>> PQ.dequeue()
        [0, 5]
        >>> PQ.is_empty()
        True
        '''
        self.size = self.size - 1           
        find_min = min(self._items)         
        self._items.remove(find_min)      
        return find_min

    def is_empty(self: 'Priority_Queue') -> bool:
        '''
        Checks if the priority queue is empty

        >>> PQ = Priority_Queue()
        >>> PQ.is_empty()
        True
        >>> PQ.enqueue(0,5)
        >>> PQ.is_empty()
        False
        '''
        return self._items == []     

    def check_size(self: 'Priority_Queue') -> int:
        '''
        Checks for the number of elements stored in priority queue

        >>> PQ = Priority_Queue()
        >>> PQ.check_size()
        0
        >>> PQ.enqueue(0,5)
        >>> PQ.check_size()
        1
        '''
        return self.size                             

    # ====================
    # Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ (int, int) -> int
    Return bit number bit_num from right in byte.

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num               

def byte_to_bits(byte):
    """ (int) -> str
    Return the representation of a byte as a string of bits.

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num)) for bit_num in range(7, -1, -1)])   


def bits_to_byte(bits):
    """ (str) -> int
    Return int represented by bits, padded on right.

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) * (1 << (7 - pos))            
      for pos in range(len(bits))])


    # ====================
    # Functions for compression


def make_freq_dict(text):
    """ (bytes) -> dict of {int: int}
    Return a dictionary that maps each byte in text to its frequency.

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    dictionary = {}
    for bytes_ in text:                                          
        if bytes_ not in dictionary:                                
            dictionary[bytes_] = 1                                  
        else:
            dictionary[bytes_] = dictionary[bytes_] + 1            
    return dictionary                                             

def huffman_tree(freq_dict):
    """ (dict of {int: int}) -> HuffmanNode
    Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    PQ = Priority_Queue()                                      
    for elements in freq_dict:                                   
        PQ.enqueue(freq_dict[elements], HuffmanNode(elements))
    while PQ.check_size() > 1:                     
        first_priority = PQ.dequeue()                               
        second_priority = PQ.dequeue()                                          
        new_node = HuffmanNode(None, first_priority[1], second_priority[1])   
        PQ.enqueue(first_priority[0] + second_priority[0], new_node)        
    return PQ.dequeue()[1]                                                 

    
def get_codes(tree):
    """ (HuffmanNode) -> dict of {int: str}
    Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    return get_string(tree)                      

def get_string(tree,string = ''):                   # helper for get_codes
    """ (HuffmanNode) -> dict of {int: str}
    Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_string(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    if tree.is_leaf():                    
        return {tree.symbol : string}
    dictionary = {}                     
    dictionary.update(get_string(tree.left,string + '0'))     
    dictionary.update(get_string(tree.right,string + '1'))    
    return dictionary
                     
def number_nodes(tree):
    """ (HuffmanNode) -> NoneType
    Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    def num_final(tree: HuffmanNode, count = 0) -> int:   
        """ 
        Number internal nodes in tree according to postorder traversal; 
        start numbering at 0. 
     
        >>> left = HuffmanNode(None, HuffmanNode(None, HuffmanNode('a'), HuffmanNode('b')), HuffmanNode('c')) 
        >>> right = HuffmanNode(None, HuffmanNode('d'), HuffmanNode(None, HuffmanNode('e'), HuffmanNode('f'))) 
        >>> tree = HuffmanNode(None, left, right) 
        >>> internal_count = internal_nodes_count(tree) 
        >>> all_numbers = internal_count - 1 
        >>> tree_ = num_final(tree,all_numbers) 
        >>> tree.number 
        4 
        >>> tree.left.number 
        1 
        >>> tree.right.number 
        3 
        >>> tree.left.left.number 
        0 
        >>> tree.right.right.number 
        2 
        """
        if tree.is_leaf():
            return count
        left_count = num_final(tree.left,count)
        new_count = num_final(tree.right, left_count)
        tree.number = new_count
        return new_count + 1
    num_final(tree)
        



def avg_length(tree, freq_dict):
    """ (HuffmanNode, dict of {int : int}) -> float
    Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    dict_of_codes = get_codes(tree)                  
    total_bit_count, total_symbol_count = 0, 0                          
    for code_symbol in freq_dict:                   
        length_of_bits = len(dict_of_codes[code_symbol])
        frequency_of_symbol = freq_dict[code_symbol]
        total_bit_count = total_bit_count + (length_of_bits * frequency_of_symbol)    
        total_symbol_count = total_symbol_count + frequency_of_symbol                 
    return total_bit_count / total_symbol_count                                    

def generate_compressed(text, codes):
    """ (bytes, dict of {int: str}) -> bytes
    Return compressed form of text, using mapping in codes for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    string, list_of_bits, index_count = '', [], 0
    for byte in text:
        string = string + codes[byte]

    for i in range(0,len(string),8):
        list_of_bits.append(bits_to_byte(string[i:i+8]))
        
    return bytes(list_of_bits)

                                                    
def tree_to_bytes(tree):
    """(HuffmanNode) -> bytes

    Return a bytes representation of the Huffman tree rooted at tree.
    The representation should be based on the postorder traversal of tree.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3, None, None), HuffmanNode(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3, None, None), HuffmanNode(2, None, None))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    if tree.is_leaf():                    
        return bytes([])
    def list_result(tree: HuffmanNode) -> list:          
        '''
        Returns a list of tree's left and right data

        >>> tree = HuffmanNode(None, HuffmanNode(3, None, None), HuffmanNode(2, None, None))
        >>> number_nodes(tree)
        >>> list_result(tree)
        [0, 3, 0, 2]
        >>> left = HuffmanNode(None, HuffmanNode(3, None, None), HuffmanNode(2, None, None))
        >>> right = HuffmanNode(5)
        >>> tree = HuffmanNode(None, left, right)
        >>> number_nodes(tree)
        >>> list_result(tree)
        [0, 3, 0, 2, 1, 0, 0, 5]
        '''
        def collection(tree: HuffmanNode) -> list:      
            '''
            Precondition: tree's root is an internal node
            Returns tree's left and right node data as a list
            '''
            if not tree:
                return []
            bytes_list = []
            if tree.is_leaf():                              
                bytes_list.append(0)
                bytes_list.append(tree.symbol)
            else:
                bytes_list.append(1)                          
                bytes_list.append(tree.number)
            return bytes_list
        if not tree:
            return []
        byte_list = list_result(tree.left) + list_result(tree.right)
        return byte_list + collection(tree.left) + collection(tree.right)
    return bytes(list_result(tree))          



def num_nodes_to_bytes(tree):
    """ (HuffmanNode) -> bytes
    Return number of nodes required to represent tree, the root of a
    numbered Huffman tree.
    """
    return bytes([tree.number + 1])

def size_to_bytes(size):
    """ (int) -> bytes
    Return the size as a bytes object.

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    return size.to_bytes(4, "little")          

def compress(in_file, out_file):
    """ (str, str) -> NoneType
    Compress contents of in_file and store results in out_file.
    """
    text = open(in_file, "rb").read()                                       
    freq = make_freq_dict(text)                                             
    tree = huffman_tree(freq)                                                 
    codes = get_codes(tree)                                                   
    number_nodes(tree)                                                       
    print("Bits per symbol:", avg_length(tree, freq))                      
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +             
    size_to_bytes(len(text)))                                                
    result += generate_compressed(text, codes)                              
    open(out_file, "wb").write(result)                                     

    # ====================
    # Functions for decompression

def generate_tree_general(node_lst, root_index):
    """ (list of ReadNode, int) -> HuffmanNode
    Return the root of the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7)]
    >>> generate_tree_general(lst, 0)
    HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None))
    """
    bytes_ = node_lst[root_index]
    if bytes_.l_type == 0:
        left = HuffmanNode(bytes_.l_data)
    else:
        left = generate_tree_general(node_lst,bytes_.l_data)
    if bytes_.r_type == 0:
        right = HuffmanNode(bytes_.r_data)
    else:
        right = generate_tree_general(node_lst,bytes_.r_data)
    return HuffmanNode(None, left, right)

def generate_tree_postorder(node_lst, root_index):
    """ (list of ReadNode, int) -> HuffmanNode
    Return the root of the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7)]
    >>> generate_tree_postorder(lst, 0)
    HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None))
    """
    return postorder(node_lst, root_index)[0]                    

def postorder(node_lst: 'list of ReadNode', root_index: int) -> '[HuffmanNode, index]':    
    '''
    Return the root of the Huffman tree corresponding to node_lst[root_index]
    as well as the index of the last recorded non internal children ReadNode.
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7)]
    >>> postorder(lst, 0)[0]
    HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None))
    '''
    bytes_ = node_lst[root_index]                              
    if bytes_.r_type == 0:
        right_values = [HuffmanNode(bytes_.r_data), root_index]
    else:
        right_values = postorder(node_lst,root_index-1)
        root_index = right_values[1]
    if bytes_.l_type == 0:
        left_values = [HuffmanNode(bytes_.l_data),root_index]
    else:
        left_values = postorder(node_lst, root_index - 1)
        root_index = left_values[1]
    return [HuffmanNode(None, left_values[0], right_values[0]),root_index]
                     

def generate_uncompressed(tree, text, size):
    """ (HuffmanNode, bytes, int) -> bytes
    Use Huffman tree to decompress size bytes from text.
    """
    if size == 0:                        
        return bytes([])
    
    bit_codes = ''                                                 
    tree_symbol = tree                                           
    list_of_bytes = []
    for bytes_ in text:                                               
        bit_codes = bit_codes + byte_to_bits(bytes_)
        
    for bit in bit_codes:
        if size > len(list_of_bytes):
            if bit == '1':
                if not tree_symbol.right.is_leaf():
                    tree_symbol = tree_symbol.right
                else:          
                    list_of_bytes.append(tree_symbol.right.symbol)
                    tree_symbol = tree                                                                          
            else:                                                                  
                if not tree_symbol.left.is_leaf():
                    tree_symbol = tree_symbol.left
                else:
                    list_of_bytes.append(tree_symbol.left.symbol)
                    tree_symbol = tree
                                 
    return bytes(list_of_bytes)                                 

def bytes_to_nodes(buf):
    """ (bytes) -> list of ReadNode

    Return a list of ReadNodes corresponding to the bytes in buf.

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []               
    for i in range(0, len(buf), 4):
        l_type = buf[i]         
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst

def bytes_to_size(buf):
    """ (bytes) -> int
    Return the size corresponding to the given 4-byte 
    little-endian representation.

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")  


def uncompress(in_file, out_file):
    """ (str, str) -> NoneType
    Uncompress contents of in_file and store results in out_file.
    """
    f = open(in_file, "rb")                                   
    num_nodes = f.read(1)[0]                                  
    buf = f.read(num_nodes * 4)                                
    node_lst = bytes_to_nodes(buf)                                
    # use generate_tree_general or generate_tree_postorder here
    tree = generate_tree_postorder(node_lst, num_nodes - 1)           
    size = bytes_to_size(f.read(4))                             
    g = open(out_file, "wb")                                    
    text = f.read()                                            
    g.write(generate_uncompressed(tree, text, size))            
    return text


    # ====================
    # Other functions


def improve_tree(tree, freq_dict):
    """(HuffmanNode, dict of {int : int}) -> NoneType

    Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    >>> tree = HuffmanNode(None, HuffmanNode(None, HuffmanNode(99, None, None), HuffmanNode(100, None, None)),HuffmanNode(None, HuffmanNode(101, None, None), HuffmanNode(None, HuffmanNode(97, None, None), HuffmanNode(98, None, None))))
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    codes = get_codes(tree)                 
    freq_dict = dict(freq_dict)                
    reverse_dict = {}                   
    assign_paths = {}                        
    finalize_tree = {}                         
    for key in codes:                        
        if len(codes[key]) in reverse_dict:
            reverse_dict[len(codes[key])].append(key)
        else:
            reverse_dict[len(codes[key])] = [key]
            
    while reverse_dict:                                  
        find_min_length = min(reverse_dict)               
        codes_max_freq = max(freq_dict, key=freq_dict.get)  
        symbol = reverse_dict[find_min_length].pop()
        assign_paths[codes_max_freq] = symbol
        if reverse_dict[find_min_length] == []:
            del reverse_dict[find_min_length]
        del freq_dict[codes_max_freq]
        
    for symbol in assign_paths:                             
        replace_path = codes[assign_paths[symbol]]                  
        finalize_tree[symbol] = replace_path
    
    while finalize_tree:                            
        tree_symbol = tree
        current_path = finalize_tree.popitem()
        symbol = current_path[0]
        path = current_path[1]
        for bit in path:                              
            if bit == '0':
                tree_symbol = tree_symbol.left
            else:
                tree_symbol = tree_symbol.right
        tree_symbol.symbol = symbol                     

   

if __name__ == "__main__":
    doctest.testmod()
    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds.".format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds.".format(fname, time.time() - start))
