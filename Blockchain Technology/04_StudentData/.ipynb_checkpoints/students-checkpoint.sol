// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StudentData {
    // Structure to hold student information
    struct Student {
        uint id;
        string name;
        uint age;
    }

    // Dynamic array to store students
    Student[] public students;

    // Fallback function to accept Ether
    fallback() external payable {}

    // Function to add a student
    function addStudent(string memory _name, uint _age) public {
        uint studentId = students.length; // ID is the index in the array
        students.push(Student(studentId, _name, _age));
    }

    // Function to retrieve student information by ID
    function getStudent(uint _id) public view returns (uint, string memory, uint) {
        require(_id < students.length, "Student ID does not exist.");
        Student memory student = students[_id];
        return (student.id, student.name, student.age);
    }

    // Function to get total number of students
    function getStudentCount() public view returns (uint) {
        return students.length;
    }
}





// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StudentData {
    // Structure to hold student information
    struct Student {
        uint id;
        string name;
        uint age;
    }

    // Dynamic array to store students
    Student[] public students;

    // Event emitted when a student is added
    event StudentAdded(uint id, string name, uint age);
    event EtherReceived(address sender, uint amount);

    // Modern way to receive ETH (preferred over fallback when only receiving)
    receive() external payable {
        emit EtherReceived(msg.sender, msg.value);
    }

    // Fallback (kept for compatibility and to satisfy requirement)
    fallback() external payable {
        emit EtherReceived(msg.sender, msg.value);
    }

    // Function to add a student
    function addStudent(string memory _name, uint _age) public {
        require(bytes(_name).length > 0, "Name cannot be empty");
        require(_age > 0 && _age < 150, "Invalid age");

        uint studentId = students.length;
        students.push(Student(studentId, _name, _age));
        emit StudentAdded(studentId, _name, _age);
    }

    // Function to retrieve student by ID
    function getStudent(uint _id) public view returns (uint, string memory, uint) {
        require(_id < students.length, "Student ID does not exist.");
        Student memory student = students[_id];
        return (student.id, student.name, student.age);
    }

    // Get total number of students
    function getStudentCount() public view returns (uint) {
        return students.length;
    }

    // View contract balance (useful for gas observation)
    function getContractBalance() public view returns (uint) {
        return address(this).balance;
    }
}
